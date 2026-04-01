//===- UndefinedMemoryBehavior.cpp - instrument seq.firmem collisions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_UNDEFINEDMEMORYBEHAVIOR
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;
using namespace hw;

namespace {

struct HashMapInstances {
    // The memory instance itself
    seq::FirMemOp memOp;
    // All the read, write, and readwrite operations that are using this memory instance
    llvm::SmallVector<seq::FirMemReadOp, 4> reads;
    llvm::SmallVector<seq::FirMemWriteOp, 4> writes;
    llvm::SmallVector<seq::FirMemReadWriteOp, 4> readWrites;
};

    /// Enables undefined memory behavior by randomly choosing what will occur with a input node.
struct UndefinedMemoryBehaviorPass
    : public circt::seq::impl::UndefinedMemoryBehaviorBase<UndefinedMemoryBehaviorPass> {
    void runOnOperation() override;   
    }// namespace

}
void UndefinedMemoryBehaviorPass::runOnOperation() {
    auto module = getOperation();

    // Set up hashmap for the SRAM instances
    llvm::SmallDenseMap<Value, HashMapInstances> sramMap;

    // Initialize the SRAM hashmap
    module.walk([&](seq::FirMemOp op) {
        Value sramResult = op.getResult();
        sramMap[sramResult].memOp = op;

        // Find all the read, write, and readwrite operations that are using this memory and add them to the hashmap at the same key
        for (Operation *user : sramResult.getUsers()) {
            llvm::TypeSwitch<Operation *>(user)
                .Case<seq::FirMemReadOp>([&](seq::FirMemReadOp readOp) {sramMap[sramResult].reads.push_back(readOp);})
                .Case<seq::FirMemWriteOp>([&](seq::FirMemWriteOp writeOp) {sramMap[sramResult].writes.push_back(writeOp);})
                .Case<seq::FirMemReadWriteOp>([&](seq::FirMemReadWriteOp readWriteOp) {sramMap[sramResult].readWrites.push_back(readWriteOp);});
        }

    });

    for (auto &object : sramMap) {
        HashMapInstances instance = object.second;
        auto readOps = instance.reads;
        auto writeOps = instance.writes;
        int randomCounter = 0;

        // If either list is empty we can return early
        if (readOps.empty() || (writeOps.empty() && readWrites.empty())) {
            continue;
        }

        // Loop through all the read and write ports and check if they are accessing the same address
        for (auto readOp : readOps) {
            ImplicitLocOpBuilder b(readOp.getLoc(), readOp);
            b.setInsertionPointAfter(readOp);
            // Maintain a list of the actual collisions that we can later use in the mux. 
            SmallVector<Value, 4> collisionList;

            for (auto writeOp : writeOps) {
                auto isSameAddress = b.create<comb::ICmpOp>(comb::ICmpPredicate::eq, readOp.getAddress(), writeOp.getAddress());

                // If they are the same address, we need to ensure they are going to collide
                Value readIsEnabled = readOp.getEnable();
                Value writeIsEnabled = writeOp.getEnable();
                Value readAndWriteEnabled = b.create<comb::AndOp>(readIsEnabled, writeIsEnabled);
                Value isCollision = b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);

                // Add this collision to the list of collisions for this read operation
                collisionList.push_back(isCollision);
            }
            
            // Check the ReadWrite ports as well
            for (auto readWriteOp : readWriteOps) {
                auto isSameAddress = b.create<comb::ICmpOp>(comb::ICmpPredicate::eq, readOp.getAddress(), readWriteOp.getAddress());

                // If they are the same address, we need to ensure they are going to collide
                Value readIsEnabled = readOp.getEnable();
                Value writeIsEnabled = writeOp.getEnable();
                Value readAndWriteEnabled = b.create<comb::AndOp>(readIsEnabled, writeIsEnabled);
                Value isCollision = b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);

                // Add this collision to the list of collisions for this read operation
                collisionList.push_back(isCollision);
            }

            if (collisionList.empty()) {
                continue;
            }

            // Use createOrFold in case there is only one collision to avoid unnecessary logic
            Value conflictTrue = b.createOrFold<comb::OrOp>(collisionList);

            // Create a random value to use for undefined behavior. We tell the solver to randomly choose a value. 
            std::string randomName = "randomValueForUndefinedBehavior" + std::to_string(randomCounter++);
            auto randomPair = module.appendInput(randomName, readOp.getType());
            Value randomVal = randomPair.second;

            // If true, we have a read-write collision and we can enable undefined memory behavior
            // This mux chooses between the correct value and an undefined value based on whether there is a collision or not
            Value mux = b.create<comb::MuxOp>(conflictTrue, randomVal, readOp.getResult());

            Operation *muxOp = mux.getDefiningOp();
            readOp.getResult().replaceAllUsesExcept(mux, muxOp);
            
        }
    }
}

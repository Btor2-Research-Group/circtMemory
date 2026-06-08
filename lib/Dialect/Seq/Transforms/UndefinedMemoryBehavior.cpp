//===- UndefinedMemoryBehavior.cpp - .Handle Undefined Behavior -----------===//
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
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Namespace.h"
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
using namespace mlir;
using namespace llvm;
// Namespace: scope for structures & functions

// Already used: can comment out
// using namespace seq;
// using namespace hw;

namespace {

struct RWMap {
  // The memory instance itself
  seq::FirMemOp memOp;
  // All the read, write, and readwrite operations that are using this memory
  // instance
  llvm::SmallVector<seq::FirMemReadOp, 4> reads;
  llvm::SmallVector<seq::FirMemWriteOp, 4> writes;
  llvm::SmallVector<seq::FirMemReadWriteOp, 4> readWrites;
};

/// Lowers seq.compreg.ce to a seq.compreg with the clock enable signal
/// built into the next logic, i.e. `next := mux(clock_enable, next, current)`
struct UndefinedMemoryBehavior
    : seq::impl::UndefinedMemoryBehaviorBase<UndefinedMemoryBehavior> {
public:
  void runOnOperation() override;

private:
  Namespace symbolNamespace;
};

} // namespace

void UndefinedMemoryBehavior::runOnOperation() {
  auto module = getOperation();

  // Set up hashmap for the SRAM instances
  // TODO : switch to
  // SmallDenseMap<Value, SmallDenseMap>
  llvm::SmallDenseMap<Value, RWMap> sramMap;

  // Initialize the SRAM hashmap
  module.walk([&](seq::FirMemOp op) {
    // Sram Result is a SSA value
    // representing a memory operation.

    // Instance of a syntactic object
    // THE memory, not just a pointer. Physical block
    Value sramResult = op.getResult();
    sramMap[sramResult].memOp = op;

    // Find all the read, write, and readwrite operations that are using this
    // memory and add them to the hashmap at the same key
    for (Operation *user : sramResult.getUsers()) {
      llvm::TypeSwitch<Operation *>(user)
          .Case<seq::FirMemReadOp>([&](seq::FirMemReadOp readOp) {
            sramMap[sramResult].reads.push_back(readOp);
          })
          .Case<seq::FirMemWriteOp>([&](seq::FirMemWriteOp writeOp) {
            sramMap[sramResult].writes.push_back(writeOp);
          })
          .Case<seq::FirMemReadWriteOp>(
              [&](seq::FirMemReadWriteOp readWriteOp) {
                sramMap[sramResult].readWrites.push_back(readWriteOp);
              })
          .Default([](auto) {});
    }
  });

  // TODO: Switch to OpBuilder
  // mlir::
  ImplicitLocOpBuilder b(module.getLoc(), module.getBody());

  //----------------
  // Initialization complete.

  // Iterate through all keys
  for (auto &object : sramMap) {
    // Iterate through the hashmaps of each key
    RWMap instance = object.second;
    auto readOps = instance.reads;
    auto writeOps = instance.writes;
    auto readWriteOps = instance.readWrites;

    // If either list is empty we can return early
    if (readOps.empty() || (writeOps.empty() && readWriteOps.empty())) {
      continue;
    }

    // Loop through all the read and write ports and check if they are accessing
    // the same address
    for (auto readOp : readOps) {
      // TODO: check
      b.setInsertionPointAfter(readOp);

      // Out of Bounds checker
      Value currentResult = readOp.getResult();
      // Width of the address?
      // Width greater than supported?
      uint64_t depth = instance.memOp.getMemory().getType().getDepth();

      // Check if empty.
      if (depth > 0) {
        Value addr = readOp.getAddress();
        Value depthValue = b.create<hw::ConstantOp>(addr.getType(), depth);
        

        // Hazard if: (Address >= Depth) which means we are out of bounds and
        // can have undefined behavior Use a symbolic value so at runtime the
        // value is chosen nondeterministically
        Value isOutOfBounds = b.createOrFold<comb::ICmpOp>(
            comb::ICmpPredicate::uge, addr, depthValue);

        
        // Randomize if needed
        auto oobName = symbolNamespace.newName("randomValueForOOB");

        // Aka choice, but used differently in application
        auto randomSymbolicOOB = verif::SymbolicValueOp::create(
            b, currentResult.getType(), b.getStringAttr(oobName));
        Value randomOOBVal = randomSymbolicOOB.getResult();

        Value muxForOOB =
            b.create<comb::MuxOp>(isOutOfBounds, randomOOBVal, currentResult);
        Operation *muxOOBOp = muxForOOB.getDefiningOp();

          /*
         Value muxForOOB =
            b.create<comb::MuxOp>(isOutOfBounds, randomOOBVal, currentResult);
        Operation *muxOOBOp = muxForOOB.getDefiningOp();
        */
        currentResult.replaceAllUsesExcept(muxForOOB, muxOOBOp);

        // Update currentResult so later logic uses the OOB-protected value.
        currentResult = muxForOOB;
      }

      // Maintain a list of the actual collisions that we can later use in the
      // mux.
      SmallVector<Value, 16> collisionList;

      for (auto writeOp : writeOps) {
        auto isSameAddress = b.create<comb::ICmpOp>(
            comb::ICmpPredicate::eq, readOp.getAddress(), writeOp.getAddress());

        // If they are the same address, we need to ensure they are going to
        // collide
        Value readIsEnabled = readOp.getEnable();
        Value writeIsEnabled = writeOp.getEnable();
        Value readAndWriteEnabled =
            b.create<comb::AndOp>(readIsEnabled, writeIsEnabled);
        Value isCollision =
            b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);

        // Add this collision to the list of collisions for this read operation
        collisionList.push_back(isCollision);
      }

      // Check the ReadWrite ports as well
      for (auto readWriteOp : readWriteOps) {
        auto isSameAddress = b.createOrFold<comb::ICmpOp>(
            comb::ICmpPredicate::eq, readOp.getAddress(),
            readWriteOp.getAddress());

        // If they are the same address, we need to ensure they are going to
        // collide
        Value readIsEnabled = readOp.getEnable();
        Value writeIsEnabled = readWriteOp.getEnable();
        Value readAndWriteEnabled =
            b.create<comb::AndOp>(readIsEnabled, writeIsEnabled);
        Value isCollision =
            b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);

        // Add this collision to the list of collisions for this read operation
        collisionList.push_back(isCollision);
      }

      if (collisionList.empty()) {
        continue;
      }

      // Use createOrFold in case there is only one collision to avoid
      // unnecessary logic
      Value conflictTrue = b.createOrFold<comb::OrOp>(mlir::ValueRange(collisionList), false);

      // Random name creation
      auto symbolicName =
          symbolNamespace.newName("randomValueForUndefinedBehavior");
      // Aka choice, but used differently in application
      auto randomSymbolicName = verif::SymbolicValueOp::create(
          b, currentResult.getType(), b.getStringAttr(symbolicName));
      Value randomVal = randomSymbolicName.getResult();

      // If true, we have a read-write collision and we can enable undefined
      // memory behavior This mux chooses between the correct value and an
      // undefined value based on whether there is a collision or not This also
      // upholds the OOB and return the random OOB value if we are out of bounds
      // regardless of collisions
      Value mux = b.create<comb::MuxOp>(conflictTrue, randomVal, currentResult);

      Operation *muxOp = mux.getDefiningOp();
      readOp.getResult().replaceAllUsesExcept(mux, muxOp);
    }

    // Write-Write Conflict: go through all the writes, see if they are the same
    // address

    // Loop through all the read and write ports and check if they are accessing
    // the same address

    /*
    for (auto writeOp : writeOps) {
      ImplicitLocOpBuilder b(module.getLoc(), module.getBody());
      b.setInsertionPointAfter(writeOp);

      // Out of Bounds checker
      Value currentResult = writeOp.getResult();
      // Width of the address?
      // Width greater than supported?
      uint64_t depth = instance.memOp.getMemory().getType().getDepth();

      // Check if incorrect address

      // I don't think this is needed for writeOp
      // however, I think it is important to check if it is out of bounds writes
      // but what do we do then?
      // TODO

      if (depth > 0) {
        Value addr = writeOp.getAddress();
        Value depthValue = b.create<hw::ConstantOp>(addr.getType(), depth);

        // Hazard if: (Address >= Depth) which means we are out of bounds and
        // can have undefined behavior Use a symbolic value so at runtime the
        // value is chosen nondeterministically
        Value isOutOfBounds = b.createOrFold<comb::ICmpOp>(
            comb::ICmpPredicate::uge, addr, depthValue);

        // Randomize if needed
        auto oobName = symbolNamespace.newName("randomValueForOOBWrite");

        // Aka choice, but used differently in application
        auto randomSymbolicOOB = verif::SymbolicValueOp::create(
            b, currentResult.getType(), b.getStringAttr(oobName));
        Value randomOOBVal = randomSymbolicOOB.getResult();

        Value muxForOOB =
            b.create<comb::MuxOp>(isOutOfBounds, randomOOBVal, currentResult);
        Operation *muxOOBOp = muxForOOB.getDefiningOp();
        currentResult.replaceAllUsesExcept(muxForOOB, muxOOBOp);

        // Update currentResult so later logic uses the OOB-protected value.
        currentResult = muxForOOB;
      }

      // Maintain a list of the actual collisions that we can later use in the
      // mux.
      SmallVector<Value, 16> collisionList;

      // Start at the write in the list AFTEr the current write, so no repeats.
      // TODO: how does this look?
      auto startIt = llvm::find(writeOps, writeOp);

      for (auto it = startIt; it != writeOps.end(); ++it) {
        auto writeOp2 = *it;
        auto isSameAddress =
            b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
                                   writeOp.getAddress(), writeOp2.getAddress());

        // If they are the same address, we need to ensure they are going to
        // collide
        Value writeIsEnabled = writeOp.getEnable();
        Value write2IsEnabled = writeOp2.getEnable();
        Value bothWritesEnabled =
            b.create<comb::AndOp>(writeIsEnabled, write2IsEnabled);
        Value isCollision =
            b.create<comb::AndOp>(isSameAddress, bothWritesEnabled);

        // Add this collision to the list of collisions for this write operation
        collisionList.push_back(isCollision);
      }

      // Check the ReadWrite ports as well
      for (auto readWriteOp : readWriteOps) {
        auto isSameAddress = b.createOrFold<comb::ICmpOp>(
            comb::ICmpPredicate::eq, readOp.getAddress(),
            readWriteOp.getAddress());

        // If they are the same address, we need to ensure they are going to
        // collide
        Value readIsEnabled = readOp.getEnable();
        Value writeIsEnabled = readWriteOp.getEnable();
        Value readAndWriteEnabled =
            b.create<comb::AndOp>(readIsEnabled, writeIsEnabled);
        Value isCollision =
            b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);

        // Add this collision to the list of collisions for this read operation
        collisionList.push_back(isCollision);
      }

      if (collisionList.empty()) {
        continue;
      }

      // I'm not sure how we make a random "write" value. How do we write
      // Use createOrFold in case there is only one collision to avoid
      // unnecessary logic
      Value conflictTrue =
          b.createOrFold<comb::OrOp>(mlir::ValueRange(collisionList), false);

      // Random name creation
      auto symbolicName =
          symbolNamespace.newName("randomValueForUndefinedBehaviorWrite");
      // Aka choice, but used differently in application
      auto randomSymbolicName = verif::SymbolicValueOp::create(
          b, currentResult.getType(), b.getStringAttr(symbolicName));
      Value randomVal = randomSymbolicName.getResult();

      // If true, we have a read-write collision and we can enable undefined
      // memory behavior This mux chooses between the correct value and an
      // undefined value based on whether there is a collision or not This also
      // upholds the OOB and return the random OOB value if we are out of bounds
      // regardless of collisions
      Value mux = b.create<comb::MuxOp>(conflictTrue, randomVal, currentResult);

      Operation *muxOp = mux.getDefiningOp();
      writeOp.getResult().replaceAllUsesExcept(mux, muxOp);
    } */
  }
}

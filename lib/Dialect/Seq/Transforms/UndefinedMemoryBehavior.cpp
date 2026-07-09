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

// DO NOT NEED
// #include "circt/Dialect/HW/HWTypes.h"
// #include "circt/Dialect/Verif/VerifDialect.h"
// #include "circt/Dialect/Verif/VerifPasses.h"
// #include "circt/Dialect/LTL/LTLTypes.h"
// #include "mlir/Transforms/DialectConversion.h"

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

  MLIRContext &ctxt = getContext(); // Avery 6.23
  ConversionTarget target(ctxt);    // Avery 6.23
  target.addLegalDialect<seq::SeqDialect, hw::HWDialect,
                         comb::CombDialect>(); // Avery 6.23

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
  // OpBuilder b(module); // Avery 6/23/26, imitating combine assert like

  //----------------
  // Initialization complete.

  // Iterate through all keys
  for (auto &object : sramMap) {
    // Iterate through the hashmaps of each key
    RWMap instance = object.second;
    auto readOps = instance.reads;
    auto writeOps = instance.writes;
    auto readWriteOps = instance.readWrites;

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

      llvm::SmallPtrSet<mlir::Operation *, 1> readExceptions;

      Value muxForOOB;
      // Check if empty.
      if (depth > 0) {
        Value addr = readOp.getAddress();
        Value depthValue = b.create<hw::ConstantOp>(addr.getType(), depth);

        // Hazard if: (Address >= Depth) which means we are out of bounds and
        // can have undefined behavior Use a symbolic value so at runtime the
        // value is chosen nondeterministically
        Value isOutOfBounds =
            b.create<comb::ICmpOp>(comb::ICmpPredicate::uge, addr, depthValue);
        // comb::ICmpOp isOutOfBounds = comb::ICmpOp::create(b,
        // comb::ICmpPredicate::gt, addr, depthValue);

        // Randomize if needed
        auto oobName = symbolNamespace.newName("randomValueForOOB");

        // Aka choice, but used differently in application
        auto randomSymbolicOOB = verif::SymbolicValueOp::create(
            b, currentResult.getType(), b.getStringAttr(oobName));
        Value randomOOBVal = randomSymbolicOOB.getResult();

        muxForOOB =
            b.create<comb::MuxOp>(isOutOfBounds, randomOOBVal, currentResult);

        Operation *muxOOBOp = muxForOOB.getDefiningOp(); // Avery 6/27

        readExceptions.insert(muxOOBOp);
        // If out of bounds, random value is the only state needed for
        //

        currentResult.replaceAllUsesExcept(muxForOOB, readExceptions);
        // Update currentResult so later logic uses the OOB-protected value.
        currentResult = muxForOOB;
        // readExceptions.insert(muxForOOB);
      }

      // If either list is empty we can return early.
      if (readOps.empty() || (writeOps.empty() && readWriteOps.empty())) {
        continue;
      }

      // Maintain a list of the actual collisions that we can later use in the
      // mux.

      // APPEND THE CURRENT RESULT TO THE

      // -------------------------------------------------------------- //
      // Check for read/write conflicts

      SmallVector<Value> collisionList;

      // SmallVector<Value> possibleValues;

      // Add the readenable to the list of collisions.

      // writeValues.insert();

      // Value oldValue = readOp.getResult();   // or memory read base
      // Value writeData = writeOp.getData();
      for (auto writeOp : writeOps) {

        b.setInsertionPointAfter(writeOp);

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

        // TODO: Push the write value on a list
        //  possibleValues.push_back(writeOp.getData()); // Avery 6/27
      }

      // Check the ReadWrite ports as well
      for (auto readWriteOp : readWriteOps) {

        b.setInsertionPointAfter(readWriteOp);

        auto isSameAddress =
            b.create<comb::ICmpOp>(comb::ICmpPredicate::eq, readOp.getAddress(),
                                   readWriteOp.getAddress());

        // If they are the same address, we need to ensure they are going to
        // collide
        Value readIsEnabled = readOp.getEnable();
        Value writeModeIsEnabled = readWriteOp.getMode(); // Avery 6/27
        Value writeIsEnabled = readWriteOp.getEnable();
        Value readWrite_ActiveWrite =
            b.create<comb::AndOp>(writeModeIsEnabled, writeIsEnabled);
        Value readAndWriteEnabled =
            b.create<comb::AndOp>(readIsEnabled, readWrite_ActiveWrite);
        Value isCollision =
            b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);

        // Add this collision to the list of collisions for this read operation
        collisionList.push_back(isCollision);

        // possibleValues.push_back(readWriteOp.getData()); // Avery 6/27
      }

      if (collisionList.empty()) {
        continue;
      }

      // Use createorFold in case there is only one collision to avoid
      // unnecessary logic
      // Value conflictTrue =
      // b.createOrFold<comb::OrOp>(mlir::ValueRange(collisionList), false);

      Value conflictTrue = b.create<comb::OrOp>(mlir::ValueRange(collisionList),
                                                false); // Avery, 6/14

      // // Add the read enable to the list.
      // collisionList.push_back(readEnable);
      // possibleValues.push_back(readOp.getResult()); //Avery 6/27

      // Random name creation
      auto symbolicName =
          symbolNamespace.newName("randomValueForReadWriteConf");
      // Aka choice, but used differently in application

      // Randomly choose one of the values (all read results/all write results)
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

      currentResult.replaceAllUsesExcept(mux, muxOp);
    }

    // TODO: 7/6 Garbage for write - write conflicts, change the %data

    // Check Write-WRite Conflicts

    // Write-Op conficts
    // Write-Write Collisions are covered

    for (auto writeOp : writeOps) {
      b.setInsertionPoint(writeOp);

      // Check OOB
      // Out of Bounds checker

      Value writeIsEnabled = writeOp.getEnable();

      Value currentEnable = writeIsEnabled;
      // Width of the address?
      // Width greater than supported?
      uint64_t depth = instance.memOp.getMemory().getType().getDepth();

      // Store the exceptions list (to make the MLIR clear to check)
      llvm::SmallPtrSet<mlir::Operation *, 1> writeExceptions;

      // Value readEnable = readOp.getEnable();

      SmallVector<Value> collisionList;

      Value muxForOOB;
      // Check if empty.
      if (depth > 0) {

        // TODO: Maybe just crash? verif.assert
        Value addr = writeOp.getAddress();
        Value depthValue = b.create<hw::ConstantOp>(addr.getType(), depth);

        auto i1 = b.getI1Type();
        Value constantTrue = b.create<hw::ConstantOp>(i1, 1); // TODO 7/6
        // Hazard if: (Address >= Depth) which means we are out of bounds and
        // can have undefined behavior Use a symbolic value so at runtime the
        // value is chosen nondeterministically
        Value isOutOfBounds =
            b.create<comb::ICmpOp>(comb::ICmpPredicate::uge, addr, depthValue);

        // Negate
        // Value not_OOB = b.create<comb::ParityOp>(isOutOfBounds); // MAKE XOR
        // 7/6
        Value not_OOB = b.create<comb::XorOp>(isOutOfBounds,
                                              constantTrue); // ADD TO MLIR 7/6

        Value write_enabled = b.create<comb::AndOp>(not_OOB, writeIsEnabled);

        // writeOp.getEnableMutable().assign(write_enabled);
        // Operation *enableOP = write_enabled.getDefiningOp(); //Avery 6/27

        // TODO: 7/6 verif::AssertOp(write_enabled)  , instead of changing the
        // enable
        // tatic AssertOp create(::mlir::ImplicitLocOpBuilder &builder,
        // ::mlir::Value property, /*optional*/::mlir::Value enable,
        // /*optional*/::mlir::StringAttr label);
        b.create<verif::AssertOp>(
            write_enabled, Value(),
            b.getStringAttr("write_enable")); // instead of changing the enable
        // If out of bounds, random value is the only state needed for
        //

        // enableExceptions.insert(enableOP);
        // currentEnable.replaceAllUsesExcept(write_enabled, enableOP);
        //  Update currentResult so later logic uses the OOB-protected value.
        //  currentEnable = write_enabled;
        // readExceptions.insert(muxForOOB);
      }

      // Check for Write-Write Conflicts

      // auto startIt = llvm::find(writeOps, writeOp);
      // auto it = startIt +1;
      // for (; it != writeOps.end(); it++) {
      // Create the WriteOp2
      // auto writeOp2 = *it;
      for (auto writeOp2 : writeOps) {

        if (writeOp2 == writeOp) {
          continue;
        }

        // Check for a Conflict
        auto isSameAddress =
            b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
                                   writeOp.getAddress(), writeOp2.getAddress());

        // If they are the same address, we need to ensure they are going to
        // collide
        Value write2IsEnabled = writeOp2.getEnable();
        Value bothWritesEnabled =
            b.create<comb::AndOp>(currentEnable, write2IsEnabled);
        Value isCollision =
            b.create<comb::AndOp>(isSameAddress, bothWritesEnabled);

        collisionList.push_back(isCollision);
        // TODO: 7/6 Add garbage to both %data.
      }

      for (auto readWriteOp : readWriteOps) {

        // Check for a Conflict
        auto isSameAddress = b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
                                                    writeOp.getAddress(),
                                                    readWriteOp.getAddress());

        // If they are the same address, we need to ensure they are going to
        // collide
        Value readWrite_WriteConflict_valid = b.create<comb::AndOp>(
            readWriteOp.getEnable(), readWriteOp.getMode());
        Value bothWritesEnabled =
            b.create<comb::AndOp>(currentEnable, readWrite_WriteConflict_valid);
        Value isWriteCollision =
            b.create<comb::AndOp>(isSameAddress, bothWritesEnabled);
        collisionList.push_back(isWriteCollision);

        // auto i1 = b.getI1Type();
        // Value constantTrue = b.create<hw::ConstantOp>(i1, 1); // TODO 7/6
        // Value notMode = b.create<comb::XorOp>(readWriteOp.getMode(),
        // constantTrue); // ADD TO MLIR 7/6
        /// Value invertMode = b.create<comb::ParityOp>(readWriteOp.getMode());
        /// // 1 if READ now // TODO: Remove
        // Value readWrite_Read_Conflict_valid =
        // b.create<comb::AndOp>(readWriteOp.getEnable(), notMode);
        //  Value isRWWCollision =
        //      b.create<comb::AndOp>(isSameAddress,
        //      readWrite_Read_Conflict_valid);
      }

      if (collisionList.empty())
        continue;

      // 7/6
      Value conflictTrue = b.create<comb::OrOp>(mlir::ValueRange(collisionList),
                                                false); // Avery, 6/14

      auto symbolicNameWrite =
          symbolNamespace.newName("randomValueForWriteWriteConf");
      // Aka choice, but used differently in application
      Value currentData = writeOp.getData();

      // Insert a random value if there is a conflict.
      auto randomSymbolicName = verif::SymbolicValueOp::create(
          b, currentData.getType(), b.getStringAttr(symbolicNameWrite));
      Value randomVal = randomSymbolicName.getResult();
      // mux : CONFLICT? random, else data

      Value mux = b.create<comb::MuxOp>(conflictTrue, randomVal, currentData);
      Operation *muxOp = mux.getDefiningOp();
      currentData.replaceAllUsesExcept(mux, muxOp);

      // 7.6 ^
    }

    // Read-Write Collisions Covered
    for (auto readWriteOp : readWriteOps) {
      b.setInsertionPoint(readWriteOp);

      // Check OOB
      // Out of Bounds checker

      Value enabled = readWriteOp.getEnable();
      Value writeModeEnabled = readWriteOp.getMode();
      Value writeIsEnabled = b.create<comb::AndOp>(writeModeEnabled, enabled);

      auto i1 = b.getI1Type();
      Value constantTrue = b.create<hw::ConstantOp>(i1, 1); // TODO 7/6
      Value readModeEnabled =
          b.create<comb::XorOp>(writeModeEnabled, constantTrue);
      Value readIsEnabled = b.create<comb::AndOp>(readModeEnabled, enabled);
      // Width of the address?
      // Width greater than supported?
      uint64_t depth = instance.memOp.getMemory().getType().getDepth();

      // Store the exceptions list (to make the MLIR clear to check)
      llvm::SmallPtrSet<mlir::Operation *, 1> writeExceptions;

      // Value readEnable = readOp.getEnable();

      SmallVector<Value> collisionList;

      SmallVector<Value> readCollisionList;

      Value currentResult = readWriteOp.getResult();

      Value muxForOOB;
      // Check if empty.
      if (depth > 0) {

        // TODO: Maybe just crash? verif.assert
        Value addr = readWriteOp.getAddress();
        Value depthValue = b.create<hw::ConstantOp>(addr.getType(), depth);

        // Hazard if: (Address >= Depth) which means we are out of bounds and
        // can have undefined behavior Use a symbolic value so at runtime the
        // value is chosen nondeterministically
        Value isOutOfBounds =
            b.create<comb::ICmpOp>(comb::ICmpPredicate::uge, addr, depthValue);

        // Negate
        Value not_OOB = b.create<comb::XorOp>(isOutOfBounds,
                                              constantTrue); // ADD TO MLIR 7/6

        // Write enabled and Not Out of Bounds
        Value write_enabled_NOOB =
            b.create<comb::AndOp>(not_OOB, writeIsEnabled);

        // TODO: 7/6 verif::AssertOp(write_enabled)  , instead of changing the
        // enable Assert that the read_write does not read out of bounds.
        b.create<verif::AssertOp>(
            write_enabled_NOOB, Value(),
            b.getStringAttr("write_enable")); // instead of changing the enable
        // If out of bounds, random value is the only state needed for
        //

        // Read OOB Check
        // Randomize if needed
        auto oobName = symbolNamespace.newName("randomValueForOOB");

        // Aka choice, but used differently in application
        auto randomSymbolicOOB = verif::SymbolicValueOp::create(
            b, currentResult.getType(), b.getStringAttr(oobName));

        Value randomOOBVal = randomSymbolicOOB.getResult();

        Value isEnabled_OOB_read =
            b.create<comb::AndOp>(isOutOfBounds, readIsEnabled);
        muxForOOB = b.create<comb::MuxOp>(isEnabled_OOB_read, randomOOBVal,
                                          currentResult);

        Operation *muxOOBOp = muxForOOB.getDefiningOp(); // Avery 6/27

        // readExceptions.insert(muxOOBOp);
        //  If out of bounds, random value is the only state needed for
        //

        currentResult.replaceAllUsesExcept(muxForOOB, muxOOBOp);
        // Update currentResult so later logic uses the OOB-protected value.
        currentResult = muxForOOB;
        // readExceptions.insert(muxForOOB);
      }

      // Check for Write-Write Conflicts

      // auto startIt = llvm::find(writeOps, writeOp);
      // auto it = startIt +1;
      // for (; it != writeOps.end(); it++) {
      // Create the WriteOp2
      // auto writeOp2 = *it;
      for (auto writeOp : writeOps) {

        // Check for a Conflict
        auto isSameAddress = b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
                                                    readWriteOp.getAddress(),
                                                    writeOp.getAddress());

        // If they are the same address, we need to ensure they are going to
        // collide
        Value writeOpIsEnabled = writeOp.getEnable();
        Value bothWritesEnabled =
            b.create<comb::AndOp>(writeIsEnabled, writeOpIsEnabled);
        Value isCollision =
            b.create<comb::AndOp>(isSameAddress, bothWritesEnabled);

        collisionList.push_back(isCollision);
        // TODO: 7/6 Add garbage to both %data.
      }

      for (auto readWriteOp2 : readWriteOps) {
        // Skip if same
        if (readWriteOp2 == readWriteOp) {
          continue;
        }

        // Check for a Conflict
        auto isSameAddress = b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
                                                    readWriteOp.getAddress(),
                                                    readWriteOp2.getAddress());

        // If they are the same address, we need to ensure they are going to
        // collide
        Value readWrite_WriteConflict_valid = b.create<comb::AndOp>(
            readWriteOp2.getEnable(), readWriteOp2.getMode());

        Value bothWritesEnabled = b.create<comb::AndOp>(
            writeIsEnabled, readWrite_WriteConflict_valid);
        Value isWriteCollision =
            b.create<comb::AndOp>(isSameAddress, bothWritesEnabled);
        collisionList.push_back(isWriteCollision);

        // auto i1 = b.getI1Type();
        // Value constantTrue = b.create<hw::ConstantOp>(i1, 1); // TODO 7/6
        // Value notMode = b.create<comb::XorOp>(readWriteOp.getMode(),
        // constantTrue); // ADD TO MLIR 7/6
        /// Value invertMode = b.create<comb::ParityOp>(readWriteOp.getMode());
        /// // 1 if READ now // TODO: Remove
        // Value readWrite_Read_Conflict_valid =
        // b.create<comb::AndOp>(readWriteOp.getEnable(), notMode);
        //  Value isRWWCollision =
        //      b.create<comb::AndOp>(isSameAddress,
        //      readWrite_Read_Conflict_valid);

        // Read-write collision
        // If the current is a read, check for collision.
        //

        Value readAndWriteEnabled =
            b.create<comb::AndOp>(readIsEnabled, readWrite_WriteConflict_valid);
        Value isReadCollision =
            b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);

        // Add this collision to the list of collisions for this read operation
        readCollisionList.push_back(isReadCollision);
      }

      // If both empty, continue.
      if (collisionList.empty() && readCollisionList.empty())
        continue;

      // 7/6

      // Check Write-Write Conflicts, input random if true.
      if (!collisionList.empty()) {
        Value conflictTrue =
            b.create<comb::OrOp>(mlir::ValueRange(collisionList),
                                 false); // Avery, 6/14

        auto symbolicNameWrite =
            symbolNamespace.newName("randomValueForRW_WriteWriteConf");
        // Aka choice, but used differently in application
        Value currentData = readWriteOp.getWriteData();

        // Insert a random value if there is a conflict.
        auto randomSymbolicName = verif::SymbolicValueOp::create(
            b, currentData.getType(), b.getStringAttr(symbolicNameWrite));
        Value randomVal = randomSymbolicName.getResult();
        // mux : CONFLICT? random, else data

        Value mux = b.create<comb::MuxOp>(conflictTrue, randomVal, currentData);
        Operation *muxOp = mux.getDefiningOp();
        currentData.replaceAllUsesExcept(mux, muxOp);

        // 7.6 ^
      }
      if (!readCollisionList.empty()) {

        // Read Collision

        Value conflictTrue =
            b.create<comb::OrOp>(mlir::ValueRange(readCollisionList),
                                 false); // Avery, 6/14

        // // Add the read enable to the list.
        // collisionList.push_back(readEnable);
        // possibleValues.push_back(readOp.getResult()); //Avery 6/27

        // Random name creation
        auto symbolicName =
            symbolNamespace.newName("randomValueForRW_ReadWriteConf");
        // Aka choice, but used differently in application

        // Randomly choose one of the values (all read results/all write
        // results)
        auto randomSymbolicName = verif::SymbolicValueOp::create(
            b, currentResult.getType(), b.getStringAttr(symbolicName));
        Value randomVal = randomSymbolicName.getResult();

        // If true, we have a read-write collision and we can enable undefined
        // memory behavior This mux chooses between the correct value and an
        // undefined value based on whether there is a collision or not This
        // also upholds the OOB and return the random OOB value if we are out of
        // bounds regardless of collisions

        Value mux =
            b.create<comb::MuxOp>(conflictTrue, randomVal, currentResult);
        Operation *muxOp = mux.getDefiningOp();

        currentResult.replaceAllUsesExcept(mux, muxOp);
      }
    }

    // for (auto readWriteOp : readWriteOps) {
    //   b.setInsertionPoint(readWriteOp);

    //   // Check OOB
    //   // Out of Bounds checker

    //   Value rwIsEnabled = readWriteOp.getEnable();

    //   Value currentEnable = rwIsEnabled;
    //   // Width of the address?
    //   // Width greater than supported?
    //   uint64_t depth = instance.memOp.getMemory().getType().getDepth();

    //   // Store the exceptions list (to make the MLIR clear to check)
    //   llvm::SmallPtrSet<mlir::Operation*, 1> readOOBExceptions;
    //   llvm::SmallPtrSet<mlir::Operation*, 1> writeOOBExceptions;
    //   // Value readEnable = readOp.getEnable();
    //    Value currentResult = readWriteOp.getResult();
    //    Value muxForOOB;

    //   // Check if empty.
    //   if (depth > 0) {
    //     Value addr = readWriteOp.getAddress();
    //     Value depthValue = b.create<hw::ConstantOp>(addr.getType(), depth);

    //     // Hazard if: (Address >= Depth) which means we are out of bounds and
    //     // can have undefined behavior Use a symbolic value so at runtime the
    //     // value is chosen nondeterministically
    //     Value isOutOfBounds =
    //         b.create<comb::ICmpOp>(comb::ICmpPredicate::uge, addr,
    //         depthValue);

    //     // Writing Out of Bounds: disable
    //     // Negate
    //     Value not_OOB = b.create<comb::ParityOp>(isOutOfBounds);
    //     Value write_mode_and_enabled = b.create<comb::AndOp>(rwIsEnabled,
    //     readWriteOp.getMode());
    //     writeOOBExceptions.insert(write_mode_and_enabled.getDefiningOp());
    //     Value write_enabled =
    //         b.create<comb::AndOp>(not_OOB, write_mode_and_enabled);

    //     readWriteOp.getEnableMutable().assign(write_enabled); // Update all

    //     Operation *enableOP = write_enabled.getDefiningOp();
    //     writeOOBExceptions.insert(enableOP);
    //     currentEnable.replaceAllUsesExcept(write_enabled,
    //     writeOOBExceptions); currentEnable = write_enabled;

    //     // Reading Out of Bounds

    //     // Randomize if needed
    //     b.setInsertionPointAfter(readWriteOp);

    //     auto oobName = symbolNamespace.newName("randomValueForOOB");

    //     // Aka choice, but used differently in application
    //     auto randomSymbolicOOB = verif::SymbolicValueOp::create(
    //         b, currentResult.getType(), b.getStringAttr(oobName));
    //     Value randomOOBVal = randomSymbolicOOB.getResult();

    //     Value oob_readMode = b.create<comb::AndOp>(isOutOfBounds,
    //     readWriteOp.getMode());

    //     Value muxForOOB =
    //         b.create<comb::MuxOp>(oob_readMode, randomOOBVal, currentResult);

    //     currentResult.replaceAllUsesExcept(muxForOOB,
    //     muxForOOB.getDefiningOp());
    //     // Update currentResult so later logic uses the OOB-protected value.
    //     currentResult = muxForOOB;
    //   }

    //   b.setInsertionPoint(readWriteOp);

    // Iterate through remaining RW ports and
    // check for conflict.

    // for (auto readWriteOp : readWriteOps){

    // Check for a Conflict
    //   auto isSameAddress =
    //       b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
    //                             writeOp.getAddress(),
    //                             readWriteOp.getAddress());

    //   // If they are the same address, we need to ensure they are going to
    //   // collide
    //   Value readWrite_WriteConflict_valid =
    //   b.create<comb::AndOp>(readWriteOp.getEnable(), readWriteOp.getMode());
    //   Value bothWritesEnabled =
    //       b.create<comb::AndOp>(currentEnable,
    //       readWrite_WriteConflict_valid);
    //   Value isWriteCollision =
    //       b.create<comb::AndOp>(isSameAddress, bothWritesEnabled);

    //   Value invertMode = b.create<comb::ParityOp>(readWriteOp.getMode()); //
    //   1 if READ now Value readWrite_Read_Conflict_valid =
    //   b.create<comb::AndOp>(readWriteOp.getEnable(), invertMode); Value
    //   isRWCollision =
    //       b.create<comb::AndOp>(isSameAddress,
    //       readWrite_Read_Conflict_valid);
    // }

    //}

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
      */

    // auto startIt = llvm::find(writeOps, writeOp);

    // for (auto it = writeOp; it != writeOps.end(); ++it) {

    //   auto writeOp2 = *it;
    //   auto isSameAddress =
    //       b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
    //                              writeOp.getAddress(),
    //                              writeOp2.getAddress());

    /*
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

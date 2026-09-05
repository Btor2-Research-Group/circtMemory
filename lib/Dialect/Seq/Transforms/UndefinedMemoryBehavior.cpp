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

// Lowers seq.compreg.ce to a seq.compreg with the clock enable signal
// built into the next logic, i.e. `next := mux(clock_enable, next, current)`
struct UndefinedMemoryBehavior
    : seq::impl::UndefinedMemoryBehaviorBase<UndefinedMemoryBehavior> {
public:
  void runOnOperation() override;

private:
  Namespace symbolNamespace;
};

} // namespace

// check_write_write_conflict();

void check_write_out_of_bounds(ImplicitLocOpBuilder &b, Value *isOutOfBoundsPtr,
                               Value *constantTruePtr,
                               Value *writeIsEnabledPtr) {
  Value not_OOB = b.create<comb::XorOp>(*isOutOfBoundsPtr, *constantTruePtr);
  Value write_enabled_NOOB = b.create<comb::AndOp>(not_OOB, *writeIsEnabledPtr);
  b.create<verif::AssertOp>(write_enabled_NOOB, Value(),
                            b.getStringAttr("write_enable"));
}

void check_read_write_conflict(ImplicitLocOpBuilder &b, Operation *readOpPTR,
                               Operation *writeOpPTR,
                               SmallVector<Value> *collisionList,
                               Value *RW_readIsEnabled, Value *writeEnabled,
                               Value *isCollision, Value *sameAddressPTR) {
  // Check if they are the same Address
  Value readAddr;
  Value writeAddr;
  Value readIsEnabled;
  Value isSameAddress;
  bool readOpInput = false;

  if (auto readOp = dyn_cast<seq::FirMemReadOp>(readOpPTR)) {
    readAddr = readOp.getAddress();
    readIsEnabled = readOp.getEnable();
    readOpInput = true;
  } else if (auto readWriteOp = dyn_cast<seq::FirMemReadWriteOp>(readOpPTR)) {
    readAddr = readWriteOp.getAddress();
    readIsEnabled = *RW_readIsEnabled;
  } else {
    // Incorrect input type, return.
    return;
  }

  if (auto writeOp = dyn_cast<seq::FirMemWriteOp>(writeOpPTR)) {
    writeAddr = writeOp.getAddress();
  } else if (auto readWriteOp = dyn_cast<seq::FirMemReadWriteOp>(writeOpPTR)) {
    writeAddr = readWriteOp.getAddress();
  } else {
    // Incorrect input type, return.
    return;
  }

  if (readOpInput) {
    isSameAddress =
        b.create<comb::ICmpOp>(comb::ICmpPredicate::eq, readAddr, writeAddr);
  } else {
    isSameAddress = *sameAddressPTR;
  }
  // Value readIsEnabled = readOp.getEnable();
  // Value writeIsEnabled = writeOp.getEnable();

  // if (!writeIsEnabled) { // No enable exists. Assume enabled.
  //   Value writeTrue = b.create<hw::ConstantOp>(i1, 1);
  //   writeIsEnabled = writeTrue;
  // }

  Value readAndWriteEnabled =
      b.create<comb::AndOp>(readIsEnabled, *writeEnabled);
  *isCollision = b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);

  // Add this collision to the list of collisions for this read operation
  collisionList->push_back(*isCollision);
  // return &isCollision;
}

void check_read_write_conflict(ImplicitLocOpBuilder &b, Operation *readOpPTR,
                               Operation *writeOpPTR,
                               SmallVector<Value> *collisionList,
                               Value *writeEnabled, Value *isCollision) {
  check_read_write_conflict(b, readOpPTR, writeOpPTR, collisionList, nullptr,
                            writeEnabled, isCollision, nullptr);
}

void check_read_out_of_bounds(
    ImplicitLocOpBuilder &b, Namespace &symbolNamespace, Operation *op,
    Value *currentResultPtr,
    llvm::SmallPtrSet<mlir::Operation *, 1> *readExceptions,
    Operation **lastCommand, uint64_t depth, Value *RW_readIsEnabled,
    Value *isOutOfBoundsPtr) {
  Value currentResult = *currentResultPtr;
  Value addr;
  Value isOutOfBoundsTemp;
  llvm::SmallPtrSet<mlir::Operation *, 1> readExceptionsTemp;
  if (!readExceptions)
    readExceptions = &readExceptionsTemp;

  if (depth > 0) {
    if (auto readOp = dyn_cast<seq::FirMemReadOp>(op)) {
      addr = readOp.getAddress();
      Value depthValue = b.create<hw::ConstantOp>(addr.getType(), depth);
      Value isOutOfBoundsTemp =
          b.create<comb::ICmpOp>(comb::ICmpPredicate::uge, addr, depthValue);
      isOutOfBoundsPtr = &isOutOfBoundsTemp;
    } else {
      if (auto writeOp = dyn_cast<seq::FirMemWriteOp>(op)) {
        // Not supposed to be called, write
        //  addr = writeOp.getAddress();
        return;
      } else if (auto readWriteOp = dyn_cast<seq::FirMemReadWriteOp>(op)) {
        // addr = readWriteOp.getAddress();
      }
    }

    // Value depthValue = b.create<hw::ConstantOp>(addr.getType(), depth);
    //  Hazard if: (Address >= Depth) which means we are out of bounds and
    //  can have undefined behavior Use a symbolic value so at runtime the
    //  value is chosen nondeterministically

    // We were not provided the info of our of bounds
    // if (!isOutOfBoundsPtr) {
    //   Value isOutOfBoundsTemp =
    //       b.create<comb::ICmpOp>(comb::ICmpPredicate::uge, addr, depthValue);
    //   isOutOfBoundsPtr = &isOutOfBoundsTemp;
    // }
    auto oobName = symbolNamespace.newName("randomValueForOOB");
    auto randomSymbolicOOB = verif::SymbolicValueOp::create(
        b, currentResult.getType(), b.getStringAttr(oobName));
    Value randomOOBVal = randomSymbolicOOB.getResult();

    // Set the random value
    // If readWriteOp, the control also depends on if a read command
    if (auto readWriteOp = dyn_cast<seq::FirMemReadWriteOp>(op)) {
      *isOutOfBoundsPtr =
          b.create<comb::AndOp>(*isOutOfBoundsPtr, *RW_readIsEnabled);
    }
    //
    Value muxForOOB =
        b.create<comb::MuxOp>(*isOutOfBoundsPtr, randomOOBVal, currentResult);
    Operation *muxOOBOp = muxForOOB.getDefiningOp();

    // Add the MUX to the list of ports to not update
    readExceptions->insert(muxOOBOp);

    // Update the final command added for continuity
    *lastCommand = muxOOBOp;

    // Update with new MUX value
    currentResult.replaceAllUsesExcept(muxForOOB, *readExceptions);

    *currentResultPtr = muxForOOB;
  }
}

void check_write_write_conflict(ImplicitLocOpBuilder &b, Operation *writeOp1PTR,
                                Operation *writeOp2PTR,
                                SmallVector<Value> *writeCollisionList,
                                Value *isSameAddress, Value *write1IsEnabled,
                                Value *write2IsEnabled) {
  Value bothWritesEnabled =
      b.create<comb::AndOp>(*write1IsEnabled, *write2IsEnabled);
  Value isWriteCollision =
      b.create<comb::AndOp>(*isSameAddress, bothWritesEnabled);
  writeCollisionList->push_back(isWriteCollision);
}

void check_rwOp_conflicts(ImplicitLocOpBuilder &b, Operation *readWriteOpPTR,
                          Operation *writeOpPTR,
                          SmallVector<Value> *writeCollisionList,
                          SmallVector<Value> *readCollisionList,
                          Value *writeEnabled, Value *readEnabled) {
  auto i1 = b.getI1Type(); // For constants.
  auto readWriteOp = dyn_cast<seq::FirMemReadWriteOp>(readWriteOpPTR);
  Value writeAddr;
  Value write2IsEnabled;
  Value rw2_write_valid;
  // No ReadWriteOp was input as the primary
  if (!(readWriteOp)) {
    return;
  }

  if (auto writeOp = dyn_cast<seq::FirMemWriteOp>(writeOpPTR)) {
    writeAddr = writeOp.getAddress();
    write2IsEnabled = writeOp.getEnable();
    if (!write2IsEnabled)
      write2IsEnabled = b.create<hw::ConstantOp>(i1, 1);

  } else if (auto readWriteOp2 = dyn_cast<seq::FirMemReadWriteOp>(writeOpPTR)) {
    writeAddr = readWriteOp2.getAddress();
    Value rw2_enable = readWriteOp2.getEnable();
    if (!rw2_enable)
      rw2_enable = b.create<hw::ConstantOp>(i1, 1);

    write2IsEnabled = b.create<comb::AndOp>(rw2_enable, readWriteOp2.getMode());
  } else {
    // Invalid input
    return;
  }

  Value isSameAddress = b.create<comb::ICmpOp>(
      comb::ICmpPredicate::eq, readWriteOp.getAddress(), writeAddr);
  // Functions
  check_write_write_conflict(b, readWriteOpPTR, writeOpPTR, writeCollisionList,
                             &isSameAddress, writeEnabled, &write2IsEnabled);
  // Write Write Conflict
  // Value bothWritesEnabled =
  //     b.create<comb::AndOp>(*writeEnabled, rw2_write_valid);
  // //Value isWriteCollision =
  //     b.create<comb::AndOp>(isSameAddress, bothWritesEnabled);

  // writeCollisionList->push_back(isWriteCollision);

  // Read-write collision, assume the first RW is a read.
  Value isReadCollision;

  check_read_write_conflict(b, readWriteOpPTR, writeOpPTR, readCollisionList,
                            readEnabled, &write2IsEnabled, &isReadCollision,
                            &isSameAddress);

  // Call read-write collisions, assuming the RW is a read
  // check_read_write_conflict(b,rwOpPTR, otherOpPTR, readCollisionList,
  // RW_readIsEnabled, RW_writeIsEnabled, writeEnabled)
}

void check_readOp_rw_conflicts() {}

void UndefinedMemoryBehavior::runOnOperation() {
  auto module = getOperation();

  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);
  target.addLegalDialect<seq::SeqDialect, hw::HWDialect, comb::CombDialect>();

  // Set up hashmap for the SRAM instances
  llvm::SmallDenseMap<Value, RWMap> sramMap;

  // Initialize the SRAM hashmap
  module.walk([&](seq::FirMemOp op) {
    // Sram Result is a SSA value
    // representing a memory operation.
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

    auto i1 = b.getI1Type(); // For constants.

    // Address all conflicts that can occur with
    // a Read Operation
    // Includes: Read-Write Conflict, Read OOB
    uint64_t depth = instance.memOp.getMemory().getType().getDepth();

    for (auto readOp : readOps) {

      // Initialize OpBuilder
      b.setInsertionPointAfter(readOp);
      Operation *lastOp = readOp; // Track if the read, readwrite, or write is
                                  // physically the last op added for correct
                                  // continuity of port references in the MLIR
      Operation *lastCommand = readOp;
      // Track the last command used for the builder for continuity
      // of port references.

      Value currentResult = readOp.getResult();

      // Tracking all updated ports so "replace all uses except" has a valid
      // argument
      llvm::SmallPtrSet<mlir::Operation *, 1> readExceptions;

      // Check if out of bounds.
      check_read_out_of_bounds(b, symbolNamespace, readOp, &currentResult,
                               &readExceptions, &lastCommand, depth, nullptr,
                               nullptr);

      // If either list is empty we can return early.
      if (readOps.empty() || (writeOps.empty() && readWriteOps.empty())) {
        continue;
      }

      // Store all read-write collisions.
      SmallVector<Value> collisionList;

      // Iterate through all write ports, looking for a conflict with the read.
      for (auto writeOp : writeOps) {

        b.setInsertionPointAfter(writeOp);

        // If they are the same address, we need to check they are going to
        // collide
        // auto isSameAddress = b.create<comb::ICmpOp>(
        //     comb::ICmpPredicate::eq, readOp.getAddress(),
        //     writeOp.getAddress());

        // Value readIsEnabled = readOp.getEnable();
        Value writeIsEnabled = writeOp.getEnable();
        if (!writeIsEnabled) { // No enable exists. Assume enabled.
          Value writeTrue = b.create<hw::ConstantOp>(i1, 1);
          writeIsEnabled = writeTrue;
        }

        Value isCollision;

        check_read_write_conflict(b, readOp, writeOp, &collisionList,
                                  &writeIsEnabled, &isCollision);

        if (!isCollision) {
          // Incorrect input
          return;
        }

        // Value readAndWriteEnabled =
        //     b.create<comb::AndOp>(readIsEnabled, writeIsEnabled);
        // Value isCollision =
        //     b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);

        // // Order the insertion of ports based on the order the ports
        // are instantiated
        if (lastOp->isBeforeInBlock(writeOp)) {
          lastCommand = isCollision.getDefiningOp();
          lastOp = writeOp;
        }

        // // Add this collision to the list of collisions for this read
        // operation collisionList.push_back(isCollision);
      }

      // Check the check the readWrite Ports for a write conflict with the
      // readOp.
      for (auto readWriteOp : readWriteOps) {

        b.setInsertionPointAfter(readWriteOp);

        // If they are the same address, we need to check if they are going to
        // collide
        // auto isSameAddress =
        //     b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
        //     readOp.getAddress(),
        //                            readWriteOp.getAddress());
        // Value readIsEnabled = readOp.getEnable();
        Value writeModeIsEnabled =
            readWriteOp.getMode(); // check for write active
        Value writeIsEnabled = readWriteOp.getEnable();

        if (!writeIsEnabled) { // No enable exists. Assume enabled.
          Value writeTrue = b.create<hw::ConstantOp>(i1, 1);
          writeIsEnabled = writeTrue;
        }

        Value rw_ActiveWrite =
            b.create<comb::AndOp>(writeModeIsEnabled, writeIsEnabled);

        Value isCollision;
        check_read_write_conflict(b, readOp, readWriteOp, &collisionList,
                                  &rw_ActiveWrite, &isCollision);
        if (!isCollision) {
          // Incorrect input

          return;
        }
        // Value readAndWriteEnabled =
        //     b.create<comb::AndOp>(readIsEnabled, readWrite_ActiveWrite);
        // Value isCollision =
        //     b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);

        // Order the insertion of ports based on the order the ports
        // are instantiated
        if (lastOp->isBeforeInBlock(readWriteOp)) {
          lastCommand = isCollision.getDefiningOp();
          lastOp = readWriteOp;
        }

        // Add this collision to the list of collisions for this read operation
        // collisionList.push_back(isCollision);
      }

      // Skip if potential collisions exist
      if (collisionList.empty()) {
        continue;
      }

      // Update the read port output to be random if a collision exists
      b.setInsertionPointAfter(lastCommand);

      Value conflictTrue =
          b.create<comb::OrOp>(mlir::ValueRange(collisionList), false);

      // Random name creation
      auto symbolicName =
          symbolNamespace.newName("randomValueForReadWriteConf");

      // Randomly choose a value (all read results/all write results)
      auto randomSymbolicName = verif::SymbolicValueOp::create(
          b, currentResult.getType(), b.getStringAttr(symbolicName));
      Value randomVal = randomSymbolicName.getResult();

      // If true, we have a read-write collision and we can enable undefined
      // memory behavior This mux chooses between the correct value and an
      // undefined value based on whether there is a collision or not. This also
      // upholds the OOB and return the random OOB value if we are out of bounds
      // regardless of collisions
      Value mux = b.create<comb::MuxOp>(conflictTrue, randomVal, currentResult);
      Operation *muxOp = mux.getDefiningOp();

      currentResult.replaceAllUsesExcept(mux, muxOp);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Address all conflicts that can occur with
    // a Write Operation
    // Includes: Write OOB, Write - Write Conflict, Write - Read Write (writing)
    // conflict
    for (auto writeOp : writeOps) {
      // Initialize OpBuilder Location
      b.setInsertionPoint(writeOp);

      Operation *lastOp = writeOp; // Track if the read, readwrite, or write is
                                   // physically the last op added for correct
                                   // continuity of port references in the MLIR

      Value writeIsEnabled = writeOp.getEnable();
      if (!writeIsEnabled) { // No enable exists. Assume enabled.
        Value writeTrue = b.create<hw::ConstantOp>(i1, 1);
        writeIsEnabled = writeTrue;
      }

      Value currentEnable = writeIsEnabled;
      uint64_t depth = instance.memOp.getMemory().getType().getDepth();

      // Track all write-write collisions
      SmallVector<Value> collisionList;

      // Check if Out of Bounds. Assert this is not the case.
      if (depth > 0) {

        Value addr = writeOp.getAddress();
        Value depthValue = b.create<hw::ConstantOp>(addr.getType(), depth);
        Value constantTrue = b.create<hw::ConstantOp>(i1, 1);

        // // Hazard if: (Address >= Depth) which means we are out of bounds and
        // // can have undefined behavior Use a symbolic value so at runtime the
        // // value is chosen nondeterministically
        Value isOutOfBounds =
            b.create<comb::ICmpOp>(comb::ICmpPredicate::uge, addr, depthValue);

        // // Assert that the write is enabled, and is in bounds
        // Value not_OOB = b.create<comb::XorOp>(isOutOfBounds, constantTrue);

        check_write_out_of_bounds(b, &isOutOfBounds, &constantTrue,
                                  &writeIsEnabled);
      }

      // Check for Write-Write Conflicts
      for (auto writeOp2 : writeOps) {

        // Skip if self.
        if (writeOp2 == writeOp) {
          continue;
        }

        // If they are the same address, we need to check if they are going to
        // collide
        auto isSameAddress =
            b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
                                   writeOp.getAddress(), writeOp2.getAddress());
        Value write2IsEnabled = writeOp2.getEnable();
        if (!write2IsEnabled) { // No enable exists. Assume enabled.
          Value write2True = b.create<hw::ConstantOp>(i1, 1);
          write2IsEnabled = write2True;
        }
        Value bothWritesEnabled =
            b.create<comb::AndOp>(currentEnable, write2IsEnabled);
        Value isCollision =
            b.create<comb::AndOp>(isSameAddress, bothWritesEnabled);

        // Order the insertion of ports based on the order the ports
        // are instantiated
        if (lastOp->isBeforeInBlock(writeOp2)) {
          lastOp = writeOp2;
        }

        // Add this collision to the list of collisions for this write operation
        collisionList.push_back(isCollision);
      }

      // Check for Write-ReadWrite (write) Conflicts
      for (auto readWriteOp : readWriteOps) {

        // If they are the same address, we need to check if they are going to
        // collide
        auto isSameAddress = b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
                                                    writeOp.getAddress(),
                                                    readWriteOp.getAddress());
        Value readWrite_WriteConflict_valid = b.create<comb::AndOp>(
            readWriteOp.getEnable(), readWriteOp.getMode());
        Value bothWritesEnabled =
            b.create<comb::AndOp>(currentEnable, readWrite_WriteConflict_valid);
        Value isWriteCollision =
            b.create<comb::AndOp>(isSameAddress, bothWritesEnabled);
        collisionList.push_back(isWriteCollision);

        // Order the insertion of ports based on the order the ports
        // are instantiated
        if (lastOp->isBeforeInBlock(
                readWriteOp)) { // If last block instantiated
          lastOp = readWriteOp;
        }
      }

      // Skip if no other ports exist.
      if (collisionList.empty())
        continue;

      // A write-write conflict occured. Write random data.
      Value conflictTrue =
          b.create<comb::OrOp>(mlir::ValueRange(collisionList), false);
      auto symbolicNameWrite =
          symbolNamespace.newName("randomValueForWriteWriteConf");
      Value currentData = writeOp.getData();

      // Insert a random value if there is a conflict.
      auto randomSymbolicName = verif::SymbolicValueOp::create(
          b, currentData.getType(), b.getStringAttr(symbolicNameWrite));
      Value randomVal = randomSymbolicName.getResult();

      Value mux = b.create<comb::MuxOp>(conflictTrue, randomVal, currentData);
      writeOp.getDataMutable().set(mux); // Update write data.
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Address all conflicts that can occur with
    // a Read-Write Operation
    // Includes: Write OOB, Read OOB,  ReadWrite - ReadWrite Conflict (Writes,
    // read-write), Write - ReadWrite (writing) conflict
    for (auto readWriteOp : readWriteOps) {
      // Initialize OpBuilder Location
      b.setInsertionPoint(readWriteOp);

      Operation *lastOp =
          readWriteOp; // Track if the read, readwrite, or write is
                       // physically the last op added for correct
                       // continuity of port references in the MLIR

      Value enabled = readWriteOp.getEnable();
      if (!enabled) { // If no enable, assume enabled
        Value enabledTrue = b.create<hw::ConstantOp>(i1, 1);
        enabled = enabledTrue;
      }

      // Create constants for the operation/reusable values.
      Value writeModeEnabled = readWriteOp.getMode();
      Value writeIsEnabled = b.create<comb::AndOp>(writeModeEnabled, enabled);

      Value constantTrue = b.create<hw::ConstantOp>(i1, 1);
      Value readModeEnabled =
          b.create<comb::XorOp>(writeModeEnabled, constantTrue);
      Value readIsEnabled = b.create<comb::AndOp>(readModeEnabled, enabled);

      uint64_t depth = instance.memOp.getMemory().getType().getDepth();

      // Track all the write-write collisions
      SmallVector<Value> writeCollisionList;
      // Track all the read-write collisions.
      SmallVector<Value> readCollisionList;

      Value currentResult = readWriteOp.getResult();

      // Used to track if a mux has been instantiated from the readWrite output
      Operation *organizationOp;
      // Check if out of bounds

      if (depth > 0) {

        Value addr = readWriteOp.getAddress();
        Value depthValue = b.create<hw::ConstantOp>(addr.getType(), depth);

        // Hazard if: (Address >= Depth) which means we are out of bounds and
        // can have undefined behavior  a symbolic value so at runtime the
        // value is chosen nondeterministically
        Value isOutOfBounds =
            b.create<comb::ICmpOp>(comb::ICmpPredicate::uge, addr, depthValue);

        // Check out of bounds write
        check_write_out_of_bounds(b, &isOutOfBounds, &constantTrue,
                                  &writeIsEnabled);

        b.setInsertionPointAfter(readWriteOp); // For correct MLIR ordering.

        // Check out of bounds read
        check_read_out_of_bounds(b, symbolNamespace, readWriteOp,
                                 &currentResult, nullptr, &organizationOp,
                                 depth, &readIsEnabled, &isOutOfBounds);

        b.setInsertionPoint(readWriteOp);
      }

      // Iterate through all writeOps for potential write-write conflicts.
      for (auto writeOp : writeOps) {

        // If they are the same address, we need to check if they are going to
        // collide
        auto isSameAddress = b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
                                                    readWriteOp.getAddress(),
                                                    writeOp.getAddress());
        Value writeOpIsEnabled = writeOp.getEnable();

        // No write enable was used when instantiating the port. Assume enabled.
        if (!writeOpIsEnabled) {
          // Create a new true port for every
          Value writeTrue = b.create<hw::ConstantOp>(i1, 1);
          writeOpIsEnabled = writeTrue;
        }

        // Check for a write-collision
        Value bothWritesEnabled =
            b.create<comb::AndOp>(writeIsEnabled, writeOpIsEnabled);
        Value isWriteCollision =
            b.create<comb::AndOp>(isSameAddress, bothWritesEnabled);

        // Order the insertion of ports based on the order the ports
        // are instantiated
        if (lastOp->isBeforeInBlock(writeOp)) {
          lastOp = writeOp;
        }

        writeCollisionList.push_back(isWriteCollision);

        // Check for a read-write collision
        Value readAndWriteEnabled =
            b.create<comb::AndOp>(readIsEnabled, writeOpIsEnabled);
        Value isReadCollision =
            b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);
        // Add this collision to the list of collisions for this read operation
        readCollisionList.push_back(isReadCollision);
      }

      // Iterate through all readWrite for potential write-write conflicts or
      // read-writeConflicts Only address the read-write conflict, assumeing
      // readWriteOp is the read, and readWriteOp2 is the write. readWriteOp2
      // will have its own iteration to check read-write collisions as well.
      for (auto readWriteOp2 : readWriteOps) {
        // Skip if same
        if (readWriteOp2 == readWriteOp) {
          continue;
        }

        // If they are the same address, we need to check if they are going to
        // collide
        // Value isSameAddress = b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
        //                                              readWriteOp.getAddress(),
        //                                              readWriteOp2.getAddress());
        // Value rw2IsEnabled = readWriteOp2.getEnable();
        // if (!rw2IsEnabled) { // Assume enabled if enable does not exist.
        //   Value rwTrue = b.create<hw::ConstantOp>(i1, 1);
        //   rw2IsEnabled = rwTrue;
        // }

        // Check for a write-write conflict.
        // Value readWrite_WriteConflict_valid =
        //     b.create<comb::AndOp>(rw2IsEnabled, readWriteOp2.getMode());

        // Functions

        // Value bothWritesEnabled = b.create<comb::AndOp>(
        //     writeIsEnabled, readWrite_WriteConflict_valid);
        // Value isWriteCollision =
        //     b.create<comb::AndOp>(isSameAddress, bothWritesEnabled);
        // writeCollisionList.push_back(isWriteCollision);
        check_rwOp_conflicts(b, readWriteOp, readWriteOp2, &writeCollisionList,
                             &readCollisionList, &writeIsEnabled,
                             &readIsEnabled);
        // Read-write collision
        // Value isReadCollision;

        //  // check_read_write_conflict(
        //       b, readWriteOp, readWriteOp2, &readCollisionList,
        //       &readIsEnabled, &readWrite_WriteConflict_valid,
        //       &isReadCollision, &isSameAddress);

        if (lastOp->isBeforeInBlock(readWriteOp2)) {
          lastOp = readWriteOp2;
        }
      }

      // If both lists are empty, continue.
      if (writeCollisionList.empty() && readCollisionList.empty())
        continue;

      // Check Write-Write Conflicts, set the data to random if one occurs.
      if (!writeCollisionList.empty()) {

        Value conflictTrue =
            b.create<comb::OrOp>(mlir::ValueRange(writeCollisionList), false);

        auto symbolicNameWrite =
            symbolNamespace.newName("randomValueForRW_WriteWriteConf");
        Value currentData = readWriteOp.getWriteData();

        // Insert a random value if there is a conflict.
        auto randomSymbolicName = verif::SymbolicValueOp::create(
            b, currentData.getType(), b.getStringAttr(symbolicNameWrite));
        Value randomVal = randomSymbolicName.getResult();
        Value mux = b.create<comb::MuxOp>(conflictTrue, randomVal, currentData);
        readWriteOp.getWriteDataMutable().set(mux);
      }

      // Check Read-Write Conflicts, set the data to random if one occurs.
      if (!readCollisionList.empty()) {
        // Mux if an OOB condition was met for a read.
        if (organizationOp) {
          b.setInsertionPointAfter(organizationOp);
        } else {
          b.setInsertionPointAfter(readWriteOp);
        }

        Value conflictTrue =
            b.create<comb::OrOp>(mlir::ValueRange(readCollisionList), false);
        // Random name creation
        auto symbolicName =
            symbolNamespace.newName("randomValueForRW_ReadWriteConf");

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
  }
}

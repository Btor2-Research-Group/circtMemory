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

// Provided an isOutOfBounds value, a constant true value, and a writeEnabled
// value, generate the ports needed to assert a write out of bounds does not
// occur.
void check_write_out_of_bounds(ImplicitLocOpBuilder &b, Value *isOutOfBoundsPtr,
                               Value *constantTruePtr,
                               Value *writeIsEnabledPtr) {
  Value not_OOB = b.create<comb::XorOp>(*isOutOfBoundsPtr, *constantTruePtr);
  Value write_enabled_NOOB = b.create<comb::AndOp>(not_OOB, *writeIsEnabledPtr);
  b.create<verif::AssertOp>(write_enabled_NOOB, Value(),
                            b.getStringAttr("write_enable"));
}

// Provided a writeOp or a readWriteOp in writeOpPTR, update writeAddr and
// writeIsEnabled to point to proper enable and addr locations. ReadWriteOp
// updates writeIsEnabled based on both mode and enable. If no enable exists,
// creates a constant 'true'.
void check_write_enable(ImplicitLocOpBuilder &b, Operation *writeOpPTR,
                        Value *writeAddr, Value *writeIsEnabled) {
  auto i1 = b.getI1Type(); // For constants.
  if (auto writeOp = dyn_cast<seq::FirMemWriteOp>(writeOpPTR)) {
    *writeAddr = writeOp.getAddress();
    *writeIsEnabled = writeOp.getEnable();
    if (!*writeIsEnabled)
      *writeIsEnabled = b.create<hw::ConstantOp>(i1, 1);

  } else if (auto readWriteOp = dyn_cast<seq::FirMemReadWriteOp>(writeOpPTR)) {
    *writeAddr = readWriteOp.getAddress();
    Value rw2_enable = readWriteOp.getEnable();
    if (!rw2_enable)
      rw2_enable = b.create<hw::ConstantOp>(i1, 1);

    *writeIsEnabled = b.create<comb::AndOp>(rw2_enable, readWriteOp.getMode());
  } else {
    // ReadOp or other is Input
    return;
  }
}

// Provided a readOp (which can also be a pointer to a readOp) and a writeOp
// (which can also be a pointer to a readWriteOp), address the read-write
// conflict. Assume readOpPTR is in a read operation. Does not address case if
// the readOpPTR is in write mode.
//
void check_read_write_conflict(ImplicitLocOpBuilder &b, Operation *readOpPTR,
                               Operation *writeOpPTR,
                               SmallVector<Value> *collisionList,
                               Value *RW_readIsEnabled, Value *writeEnabled,
                               Value *isCollision, Value *sameAddressPTR) {
  // Check if they are the same Address
  Value readAddr;
  Value writeAddr;
  Value readIsEnabled;
  Value isSameAddress = *sameAddressPTR;

  if (auto readOp = dyn_cast<seq::FirMemReadOp>(readOpPTR)) {
    readAddr = readOp.getAddress();
    readIsEnabled = readOp.getEnable();
  } else if (auto readWriteOp = dyn_cast<seq::FirMemReadWriteOp>(readOpPTR)) {
    readAddr = readWriteOp.getAddress();
    readIsEnabled = *RW_readIsEnabled;
  } else {
    // Incorrect input type, return.
    return;
  }

  // Check the Second input, which should be a write.
  if (auto writeOp = dyn_cast<seq::FirMemWriteOp>(writeOpPTR)) {
    writeAddr = writeOp.getAddress();
  } else if (auto readWriteOp = dyn_cast<seq::FirMemReadWriteOp>(writeOpPTR)) {
    writeAddr = readWriteOp.getAddress();
  } else {
    // Incorrect input type, return.
    return; // TODO: what do?
  }

  Value readAndWriteEnabled =
      b.create<comb::AndOp>(readIsEnabled, *writeEnabled);

  // Used in the caller to determine if output of the function needs to be
  // rearranged
  *isCollision = b.create<comb::AndOp>(isSameAddress, readAndWriteEnabled);

  // Add this collision to the list of collisions for this read operation
  collisionList->push_back(*isCollision);
  // return &isCollision;
}

// Overloaded, enough information for a readOp
// void check_read_write_conflict(ImplicitLocOpBuilder &b, Operation *readOpPTR,
//                                Operation *writeOpPTR,
//                                SmallVector<Value> *collisionList,
//                                Value *writeEnabled, Value *isCollision) {
//   check_read_write_conflict(b, readOpPTR, writeOpPTR, collisionList, nullptr,
//                             writeEnabled, isCollision, nullptr);
// }

// Check readOp and readWriteOps out of bounds reads.
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

// Check if two operations (can be either readWriteOp or writeOp, any combo)
// collide.
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

// Address all necessary conflicts between a writeOp and the other input
// operation a readOp should not be the other, as a writeOp does not change if
// it collides with a read.
void check_writeOp_conflicts(ImplicitLocOpBuilder &b, Operation *writeOpPTR,
                             Operation *otherOpPTR,
                             SmallVector<Value> *writeCollisionList,
                             Value *writeEnabled) {

  // auto i1 = b.getI1Type(); // For constants.
  auto writeOp = dyn_cast<seq::FirMemWriteOp>(writeOpPTR);
  Value writeAddr;
  Value other_writeIsEnabled; // If the other operation
  // No ReadWriteOp was input as the primary
  if (!(writeOp)) {
    return;
  }

  check_write_enable(b, otherOpPTR, &writeAddr, &other_writeIsEnabled);
  if (!other_writeIsEnabled) {
    return; // ReadOp
  }

  // isSame Address
  Value isSameAddress = b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
                                               writeOp.getAddress(), writeAddr);
  // If other is a write, give it an enable if needed
  // If other is a readWrite, give it an enable if needed/check if a write.

  check_write_write_conflict(b, writeOpPTR, otherOpPTR, writeCollisionList,
                             &isSameAddress, writeEnabled,
                             &other_writeIsEnabled);
}

// Check all possible conflicts between a readOp and another operation, assumed
// to be a write. If the second operation is a write, returns without changing
// any input pointers.
void check_readOp_conflicts(ImplicitLocOpBuilder &b, Operation *readOpPTR,
                            Operation *otherOpPTR,
                            SmallVector<Value> *readCollisionList,
                            Value *isCollision) {
  // auto i1 = b.getI1Type(); // For constants.
  Value writeAddr;
  Value other_writeIsEnabled;
  auto readOp = dyn_cast<seq::FirMemReadOp>(readOpPTR);
  // If ReadOP is not a readOp, error. Exit.
  if (!(readOp)) {
    return;
  }
  Value readEnabled = readOp.getEnable();
  // check_write_enable(ImplicitLocOpBuilder &b, Operation *writeOpPTR, Value *
  // writeAddr, Value * writeIsEnabled);

  check_write_enable(b, otherOpPTR, &writeAddr, &other_writeIsEnabled);
  if (!other_writeIsEnabled) {
    return; // ReadOp
  }

  Value isSameAddress = b.create<comb::ICmpOp>(comb::ICmpPredicate::eq,
                                               readOp.getAddress(), writeAddr);
  // check_read_write_conflict(b, readOpPTR, otherOpPTR, readCollisionList,
  //                           &other_writeIsEnabled, isCollision);

  check_read_write_conflict(b, readOpPTR, otherOpPTR, readCollisionList,
                            &readEnabled, &other_writeIsEnabled, isCollision,
                            &isSameAddress);
}

// Address all conflicts between a readWriteOp and another operation.
// If the other operation (writeOpPTR) is a readOp, returns without changing
// other variables.
void check_rwOp_conflicts(ImplicitLocOpBuilder &b, Operation *readWriteOpPTR,
                          Operation *otherOpPTR,
                          SmallVector<Value> *writeCollisionList,
                          SmallVector<Value> *readCollisionList,
                          Value *writeEnabled, Value *readEnabled) {
  // auto i1 = b.getI1Type(); // For constants.
  auto readWriteOp = dyn_cast<seq::FirMemReadWriteOp>(readWriteOpPTR);
  Value writeAddr;
  Value other_writeIsEnabled;
  // Value rw2_write_valid;
  //  No ReadWriteOp was input as the primary
  if (!(readWriteOp)) {
    return;
  }

  check_write_enable(b, otherOpPTR, &writeAddr, &other_writeIsEnabled);
  if (!other_writeIsEnabled) {
    return; // ReadOp
  }

  Value isSameAddress = b.create<comb::ICmpOp>(
      comb::ICmpPredicate::eq, readWriteOp.getAddress(), writeAddr);
  // Functions
  check_write_write_conflict(b, readWriteOpPTR, otherOpPTR, writeCollisionList,
                             &isSameAddress, writeEnabled,
                             &other_writeIsEnabled);

  Value isReadCollision; // Updated but unused. If wanting to work with the
                         // collision Value, this is where it will be.

  check_read_write_conflict(b, readWriteOpPTR, otherOpPTR, readCollisionList,
                            readEnabled, &other_writeIsEnabled,
                            &isReadCollision, &isSameAddress);
}

// Provided a list of possible conflicts, generate hardware components
// if such a conflict exists. Store the final mux generated in provided variable
// muxPTR.
void address_conflicts(ImplicitLocOpBuilder &b, Namespace &symbolNamespace,
                       Value *muxPTR, SmallVector<Value> *collisionList,
                       Value *currentResult) {

  Value conflictTrue =
      b.create<comb::OrOp>(mlir::ValueRange(*collisionList), false);
  // Random name creation
  auto symbolicName = symbolNamespace.newName("randomValueForConflict");

  // Randomly choose one of the values (all read results/all write
  // results)
  auto randomSymbolicName = verif::SymbolicValueOp::create(
      b, currentResult->getType(), b.getStringAttr(symbolicName));
  Value randomVal = randomSymbolicName.getResult();

  // If true, we have a read-write collision and we can enable undefined
  // memory behavior This mux chooses between the correct value and an
  // undefined value based on whether there is a collision or not This
  // also upholds the OOB and return the random OOB value if we are out of
  // bounds regardless of collisions
  Value mux = b.create<comb::MuxOp>(conflictTrue, randomVal, *currentResult);
  *muxPTR = mux; // Update the mux output
}

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

        Value isCollision;
        check_readOp_conflicts(b, readOp, writeOp, &collisionList,
                               &isCollision);

        if (!isCollision) {
          // Incorrect input
          return;
        }

        if (lastOp->isBeforeInBlock(writeOp)) {
          lastCommand = isCollision.getDefiningOp();
          lastOp = writeOp;
        }
      }

      // Check the check the readWrite Ports for a write conflict with the
      // readOp.
      for (auto readWriteOp : readWriteOps) {

        b.setInsertionPointAfter(readWriteOp);

        Value isCollision;
        check_readOp_conflicts(b, readOp, readWriteOp, &collisionList,
                               &isCollision);

        if (!isCollision) {
          // Error in handing the conflicts.
          return;
        }

        // Order the insertion of ports based on the order the ports
        // are instantiated
        if (lastOp->isBeforeInBlock(readWriteOp)) {
          lastCommand = isCollision.getDefiningOp();
          lastOp = readWriteOp;
        }
      }

      // Skip if potential collisions exist
      if (collisionList.empty()) {
        continue;
      }

      // Update the read port output to be random if a collision exists
      b.setInsertionPointAfter(lastCommand);

      Value mux;
      address_conflicts(b, symbolNamespace, &mux, &collisionList,
                        &currentResult);

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

      // Value currentEnable = writeIsEnabled;
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

        check_writeOp_conflicts(b, writeOp, writeOp2, &collisionList,
                                &writeIsEnabled);
      }

      // Check for Write-ReadWrite (write) Conflicts
      for (auto readWriteOp : readWriteOps) {

        // If they are the same address, we need to check if they are going to
        // collide
        check_writeOp_conflicts(b, writeOp, readWriteOp, &collisionList,
                                &writeIsEnabled);

        if (lastOp->isBeforeInBlock(
                readWriteOp)) { // If last block instantiated
          lastOp = readWriteOp;
        }
      }

      // Skip if no other ports exist.
      if (collisionList.empty())
        continue;

      // A write-write conflict occured. Write random data.
      Value mux;
      Value currentData = writeOp.getData();
      address_conflicts(b, symbolNamespace, &mux, &collisionList, &currentData);

      // Insert a random value if there is a conflict.
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

        check_rwOp_conflicts(b, readWriteOp, writeOp, &writeCollisionList,
                             &readCollisionList, &writeIsEnabled,
                             &readIsEnabled);

        if (lastOp->isBeforeInBlock(writeOp)) {
          lastOp = writeOp;
        }
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

        check_rwOp_conflicts(b, readWriteOp, readWriteOp2, &writeCollisionList,
                             &readCollisionList, &writeIsEnabled,
                             &readIsEnabled);

        if (lastOp->isBeforeInBlock(readWriteOp2)) {
          lastOp = readWriteOp2;
        }
      }

      // If both lists are empty, continue.
      if (writeCollisionList.empty() && readCollisionList.empty())
        continue;

      // Check Write-Write Conflicts, set the data to random if one occurs.
      if (!writeCollisionList.empty()) {

        Value currentData = readWriteOp.getWriteData();
        Value mux;
        address_conflicts(b, symbolNamespace, &mux, &writeCollisionList,
                          &currentData);
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

        Value mux;

        address_conflicts(b, symbolNamespace, &mux, &readCollisionList,
                          &currentResult);
        Operation *muxOp = mux.getDefiningOp();
        currentResult.replaceAllUsesExcept(mux, muxOp);
      }
    }
  }
}

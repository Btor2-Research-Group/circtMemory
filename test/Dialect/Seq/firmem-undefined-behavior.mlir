// RUN: circt-opt --undefined-memory-behavior --split-input-file %s | FileCheck %s
// circt-opt --undefined-memory-behavior tests/Dialect/Seq/firmem-undefined-behavior.mlir

// ton of binaries
// look for circt-opt

// circt-opt
// https://github.com/chipsalliance/firrtl/blob/1.6.x/src/test/scala/firrtl/backends/experimental/smt/random/UndefinedMemoryBehaviorSpec.scala

// do i check isOutOfBounds, muxForOOB as well?

 // Check that memory exists
// Depth: Needs to exist, because depth may not be constant across systems
// Check that the address is no longer within the depth (m_r_oob in scala)
// RANDOM data is chosen. Do we need this? How do we choose the "old" data - is there a vague way to to it?

// Check that out of bounds variable is active
// Check that the data is random

// NOTES: Need to do reading out of bounds for read-writes
// Read-write: mode = 0 when a read, =1 when a write

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Multiple Writes / Reads/ ReadWrites same address

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
// Write-Write Conflict
// Same Address, Different Address, one enabled one not, both enabled, both disabled


  // CHECK-LABEL : hw.module @WriteWriteConflict_BothEnabled
  // Read Out of Bounds check
  // CHECK-NEXT : [[ENABLE:%.+]] = hw.constant true
  // CHECK-NEXT : [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT : %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  // Create the write ports
  // CHECK-NEXT : seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[ENABLE]] : <12 x 20>

  // Check if the write is out of bounds


  // Declare the Second write port

  // Check that the output mux is controlled by a conflict, and is between the random and intended read
  // CHECK-NEXT : hw.output [[TMP9]] : i20
  // CHECK-NEXT : }
 
//  hw.module @WriteWriteConflict_BothEnabled(in %data: i20, in %clock: !seq.clock,  out z: i20) {
//   %enable = hw.constant true // Set to constant 1
//   %addr = hw.constant 6 : i4
//   %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

//   %0 = seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enable: <12 x 20>
//   %1 = seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enable: <12 x 20>

//   hw.output %0 : i20
//   }

// TODO: WRITE OUT OF BOUNDS : Anything? I don't think so. Nothing gets returned when you write.



//-----

//CHECK-LABEL: hw.module @WriteConflict_OOB
  // Read Out of Bounds check
  // CHECK-NEXT: [[ENABLE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // OOB Check
  // CHECK-NEXT: [[TMP0_W:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.parity [[TMP1_W]] : i1
  // CHECK-NEXT: [[NEWENABLE:%.+]] = comb.and [[TMP2_W]], [[ENABLE]] : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[NEWENABLE]] : <12 x 20>
  // CHECK-NEXT: hw.output [[ADDR]] : i4
  // CHECK-NEXT: }
  hw.module @WriteConflict_OOB(in %data: i20, in %clock: !seq.clock,  out z: i4) {
   %enable = hw.constant true // Set to constant 1
   %addr = hw.constant 6 : i4
   %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

    seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enable: <12 x 20>

   // Does a write have an output?
    hw.output %addr : i4
  }

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Read-Write, Write Conflict
// Types: Both write. Both enabled, one of each enabled, different address, out of bounds.
// Read-write is a read: all read-write collisions with a read-write/write pair.






//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Read, Read Write Conflict
// Types: read/write collisions, but for read write.
// Both Enabled, Both disabled, one enabled one not, Diff address

// Read, Read Write Both Enabled
// WORKS
//------

  // CHECK-LABEL: hw.module @Read_ReadWriteConflict_BothEnabled
  // CHECK-NEXT: [[ENABLERD:%.+]] = hw.constant true
  // CHECK-NEXT: [[ENABLEWR:%.+]] = hw.constant true
  // CHECK-NEXT: [[MODE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLERD]] : <12 x 20>
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // Is the Read-Write Enabled/In write mode?
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and [[MODE]], [[ENABLEWR]] : i1 
  // Are both Enabled?
  // CHECK-NEXT: [[TMP6:%.+]] = comb.and [[ENABLERD]], [[TMP5]] : i1 
  // Are they the same addr & enabled?
  // CHECK-NEXT: [[TMP7:%.+]] = comb.and [[TMP4]], [[TMP6]] : i1 
  // CHECK-NEXT: [[TMP8:%.+]] = comb.or [[TMP7]] : i1
  // CHECK-NEXT: [[TMP9:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP10:%.+]] = comb.mux [[TMP8]], [[TMP9]], [[TMP3]] : i20
  // CHECK-NEXT: [[WRITE:%.+]] = seq.firmem.read_write_port %mem[[[ADDR]]] = %data if [[MODE]], clock %clock enable [[ENABLEWR]] : <12 x 20>
  // CHECK-NEXT: hw.output [[TMP10]] : i20
  // CHECK-NEXT: }
  hw.module @Read_ReadWriteConflict_BothEnabled(in %data: i20, in %clock: !seq.clock, out z: i20) {
  %enableRD = hw.constant true // Set to constant 1
  %enableWR = hw.constant true // Set to constant 1

  %mode = hw.constant true // Set the ReadWrite to writing
  %addr = hw.constant 6 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enableRD: <12 x 20>
  %1 = seq.firmem.read_write_port %mem[%addr] = %data if %mode, clock %clock enable %enableWR: <12 x 20>

  hw.output %0 : i20
  }
//


// Read, Read Write, Read Enabled
// WORKS
//------

  // CHECK-LABEL : hw.module @Read_ReadWriteConflict_ReadEnabled
  // CHECK-NEXT : [[ENABLEREAD:%.+]] = hw.constant true 
  // CHECK-NEXT : [[ENABLERW:%.+]] = hw.constant false 
  // CHECK-NEXT : [[MODE:%.+]] = hw.constant true
  // CHECK-NEXT : [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT : %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT : [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLEREAD]] : <12 x 20>
  // CHECK-NEXT : [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT : [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT : [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT : [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // Read write
  // CHECK-NEXT : [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // Is the Read-Write Enabled/In write mode?
  // CHECK-NEXT : [[TMP5:%.+]] = comb.and [[MODE]], [[ENABLERW]] : i1 
  // Are both Enabled?
  // CHECK-NEXT : [[TMP6:%.+]] = comb.and [[ENABLEREAD]], [[TMP5]] : i1 
  // Are they the same addr & enabled?
  // CHECK-NEXT : [[TMP7:%.+]] = comb.and [[TMP4]], [[TMP6]] : i1 
  // CHECK-NEXT : [[TMP8:%.+]] = comb.or [[TMP7]] : i1
  // CHECK-NEXT : [[TMP9:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT : [[TMP10:%.+]] = comb.mux [[TMP8]], [[TMP9]], [[TMP3]] : i20
  // CHECK-NEXT : [[WRITE:%.+]] = seq.firmem.read_write_port %mem[[[ADDR]]] = %data if [[MODE]], clock %clock enable [[ENABLERW]] : <12 x 20>
  // CHECK-NEXT : hw.output [[TMP10]] : i20
  // CHECK-NEXT : }
  hw.module @Read_ReadWriteConflict_ReadEnabled(in %data: i20, in %clock: !seq.clock, out z: i20) {
  %enableREAD = hw.constant true // Set to constant 1
  %enableRW = hw.constant false
  %mode = hw.constant true // Set the ReadWrite to writing
  %addr = hw.constant 6 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enableREAD: <12 x 20>
  %1 = seq.firmem.read_write_port %mem[%addr] = %data if %mode, clock %clock enable %enableRW: <12 x 20>

  hw.output %0 : i20
  }
//


// Read, Read Write, Both Disabled
// WORKS
//------

  // CHECK-LABEL : hw.module @Read_ReadWriteConflict_Disabled
  // CHECK-NEXT : [[ENABLEREAD:%.+]] = hw.constant false 
  // CHECK-NEXT : [[ENABLERW:%.+]] = hw.constant false 
  // CHECK-NEXT : [[MODE:%.+]] = hw.constant true
  // CHECK-NEXT : [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT : %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT : [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLEREAD]] : <12 x 20>
  // CHECK-NEXT : [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT : [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT : [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT : [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // Read write
  // CHECK-NEXT : [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // Is the Read-Write Enabled/In write mode?
  // CHECK-NEXT : [[TMP5:%.+]] = comb.and [[MODE]], [[ENABLERW]] : i1 
  // Are both Enabled?
  // CHECK-NEXT : [[TMP6:%.+]] = comb.and [[ENABLEREAD]], [[TMP5]] : i1 
  // Are they the same addr & enabled?
  // CHECK-NEXT : [[TMP7:%.+]] = comb.and [[TMP4]], [[TMP6]] : i1 
  // CHECK-NEXT : [[TMP8:%.+]] = comb.or [[TMP7]] : i1
  // CHECK-NEXT : [[TMP9:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT : [[TMP10:%.+]] = comb.mux [[TMP8]], [[TMP9]], [[TMP3]] : i20
  // CHECK-NEXT : [[WRITE:%.+]] = seq.firmem.read_write_port %mem[[[ADDR]]] = %data if [[MODE]], clock %clock enable [[ENABLERW]] : <12 x 20>
  // CHECK-NEXT : hw.output [[TMP10]] : i20
  // CHECK-NEXT : }
  hw.module @Read_ReadWriteConflict_Disabled(in %data: i20, in %clock: !seq.clock, out z: i20) {
  %enableREAD = hw.constant false // Set to constant 1
  %enableRW = hw.constant false
  %mode = hw.constant true // Set the ReadWrite to writing
  %addr = hw.constant 6 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enableREAD: <12 x 20>
  %1 = seq.firmem.read_write_port %mem[%addr] = %data if %mode, clock %clock enable %enableRW: <12 x 20>

  hw.output %0 : i20
  }
//


// Read, Read Write - Different Address
// WORKS
//------

  // CHECK-LABEL : hw.module @Read_ReadWriteConflict_DiffAddr
  // CHECK-NEXT : [[ENABLE:%.+]] = hw.constant true 
  // CHECK-NEXT : [[MODE:%.+]] = hw.constant true
  // CHECK-NEXT : [[ADDRREAD:%.+]] = hw.constant 6 : i4
   // CHECK-NEXT : [[ADDRRW:%.+]] = hw.constant 4 : i4
  // CHECK-NEXT : %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT : [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDRREAD]]], clock %clock enable [[ENABLE]] : <12 x 20>
  // CHECK-NEXT : [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT : [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT : [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT : [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // Read write
  // CHECK-NEXT : [[TMP4:%.+]] = comb.icmp eq [[ADDRREAD]], [[ADDRRW]] : i4 
  // Is the Read-Write Enabled/In write mode?
  // CHECK-NEXT : [[TMP5:%.+]] = comb.and [[MODE]], [[ENABLE]] : i1 
  // Are both Enabled?
  // CHECK-NEXT : [[TMP6:%.+]] = comb.and [[ENABLE]], [[TMP5]] : i1 
  // Are they the same addr & enabled?
  // CHECK-NEXT : [[TMP7:%.+]] = comb.and [[TMP4]], [[TMP6]] : i1 
  // CHECK-NEXT : [[TMP8:%.+]] = comb.or [[TMP7]] : i1
  // CHECK-NEXT : [[TMP9:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT : [[TMP10:%.+]] = comb.mux [[TMP8]], [[TMP9]], [[TMP3]] : i20
  // CHECK-NEXT : [[WRITE:%.+]] = seq.firmem.read_write_port %mem[[[ADDRRW]]] = %data if [[MODE]], clock %clock enable [[ENABLE]] : <12 x 20>
  // CHECK-NEXT : hw.output [[TMP10]] : i20
  // CHECK-NEXT : }
  hw.module @Read_ReadWriteConflict_DiffAddr(in %data: i20, in %clock: !seq.clock, out z: i20) {
  %enable = hw.constant true // Set to constant 1
  %mode = hw.constant true // Set the ReadWrite to writing
  %addrREAD = hw.constant 6 : i4
  %addrRW = hw.constant 4 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addrREAD], clock %clock enable %enable: <12 x 20>
  %1 = seq.firmem.read_write_port %mem[%addrRW] = %data if %mode, clock %clock enable %enable: <12 x 20>

  hw.output %0 : i20
  }
//





//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// READ WRITE CONFLICTS
// Both Enabled, One of each enabled, out of bounds.

// 
// READ WRITE, BOTH ENABLED
// 

//------

  // CHECK-LABEL: hw.module @ReadWriteConflict_BothEnabled
  // Read Out of Bounds check
  // CHECK-NEXT: [[ENABLEREAD:%.+]] = hw.constant true
  // CHECK-NEXT: [[ENABLEWRITE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLEREAD]] : <12 x 20>
  // READ OOB
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // WRITE OOB
  // CHECK-NEXT: [[TMP0_W:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.parity [[TMP1_W]] : i1
  // CHECK-NEXT: [[NEW_ENABLEWRITE:%.+]] = comb.and [[TMP2_W]], [[ENABLEWRITE]] : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[NEW_ENABLEWRITE]] : <12 x 20>
  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLEREAD]], [[NEW_ENABLEWRITE]] : i1 
  // CHECK-NEXT: [[TMP6:%.+]] = comb.and [[TMP4]], [[TMP5]] : i1 
  // CHECK-NEXT: [[TMP7:%.+]] = comb.or [[TMP6]] : i1
  // CHECK-NEXT: [[TMP8:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP9:%.+]] = comb.mux [[TMP7]], [[TMP8]], [[TMP3]] : i20
  // Check that the output mux is controlled by a conflict, and is between the random and intended read
  // CHECK-NEXT: hw.output [[TMP9]] : i20
  // CHECK-NEXT: }
  hw.module @ReadWriteConflict_BothEnabled(in %data: i20, in %clock: !seq.clock,  out z: i20) {
  %enableRead = hw.constant true // Set to constant 1
  %enableWrite = hw.constant true // Set to constant 1
  %addr = hw.constant 6 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enableRead: <12 x 20>
  seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enableWrite: <12 x 20>

  hw.output %0 : i20
  }
//

// Read write, Read disabled
// 
//------

  // CHECK-LABEL: hw.module @ReadWriteConflict_ReadDisabled
  // CHECK-NEXT: [[ENABLEREAD:%.+]] = hw.constant false
  // CHECK-NEXT: [[ENABLEWRITE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLEREAD]] : <12 x 20>
  // READ OOB
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // WRITE OOB
  // CHECK-NEXT: [[TMP0_W:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.parity [[TMP1_W]] : i1
  // CHECK-NEXT: [[NEW_ENABLEWRITE:%.+]] = comb.and [[TMP2_W]], [[ENABLEWRITE]] : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[NEW_ENABLEWRITE]] : <12 x 20>
  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLEREAD]], [[NEW_ENABLEWRITE]] : i1 
  // CHECK-NEXT: [[TMP6:%.+]] = comb.and [[TMP4]], [[TMP5]] : i1 
  // CHECK-NEXT: [[TMP7:%.+]] = comb.or [[TMP6]] : i1
  // CHECK-NEXT: [[TMP8:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP9:%.+]] = comb.mux [[TMP7]], [[TMP8]], [[TMP3]] : i20
  // Check that the output mux is controlled by a conflict, and is between the random and intended read
  // CHECK-NEXT: hw.output [[TMP9]] : i20
  // CHECK-NEXT: }
  hw.module @ReadWriteConflict_ReadDisabled(in %data: i20, in %clock: !seq.clock,  out z: i20) {
  %enableRead = hw.constant false // Set to constant 1
  %enableWrite = hw.constant true
  %addr = hw.constant 6 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enableRead: <12 x 20>
  seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enableWrite: <12 x 20>

  hw.output %0 : i20
  }


// Read write, write disabled
// Works
//-----

  // CHECK-LABEL: hw.module @ReadWriteConflict_WriteDisabled
  // CHECK-NEXT: [[ENABLEREAD:%.+]] = hw.constant true
  // CHECK-NEXT: [[ENABLEWRITE:%.+]] = hw.constant false
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLEREAD]] : <12 x 20>
  // READ OOB
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // WRITE OOB
  // CHECK-NEXT: [[TMP0_W:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.parity [[TMP1_W]] : i1
  // CHECK-NEXT: [[NEW_ENABLEWRITE:%.+]] = comb.and [[TMP2_W]], [[ENABLEWRITE]] : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[NEW_ENABLEWRITE]] : <12 x 20>
  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLEREAD]], [[NEW_ENABLEWRITE]] : i1 
  // CHECK-NEXT: [[TMP6:%.+]] = comb.and [[TMP4]], [[TMP5]] : i1 
  // CHECK-NEXT: [[TMP7:%.+]] = comb.or [[TMP6]] : i1
  // CHECK-NEXT: [[TMP8:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP9:%.+]] = comb.mux [[TMP7]], [[TMP8]], [[TMP3]] : i20
  // Check that the output mux is controlled by a conflict, and is between the random and intended read
  // CHECK-NEXT: hw.output [[TMP9]] : i20
  // CHECK-NEXT: }
  hw.module @ReadWriteConflict_WriteDisabled(in %data: i20, in %clock: !seq.clock,  out z: i20) {
  %enableRead = hw.constant true // Set to constant 1
  %enableWrite = hw.constant false
  %addr = hw.constant 6 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enableRead: <12 x 20>
  seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enableWrite: <12 x 20>

  hw.output %0 : i20
  }


//
// Read Write, Both Enabled & Same out of bounds address
//-----

  // CHECK-LABEL: hw.module @ReadWriteConflict_BothEnabledOOB
  // CHECK-NEXT: [[ENABLEREAD:%.+]] = hw.constant true
  // CHECK-NEXT: [[ENABLEWRITE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant -3 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLEREAD]] : <12 x 20>
  // READ OOB
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // WRITE OOB
  // CHECK-NEXT: [[TMP0_W:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.parity [[TMP1_W]] : i1
  // CHECK-NEXT: [[NEW_ENABLEWRITE:%.+]] = comb.and [[TMP2_W]], [[ENABLEWRITE]] : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[NEW_ENABLEWRITE]] : <12 x 20>
  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLEREAD]], [[NEW_ENABLEWRITE]] : i1 
  // CHECK-NEXT: [[TMP6:%.+]] = comb.and [[TMP4]], [[TMP5]] : i1 
  // CHECK-NEXT: [[TMP7:%.+]] = comb.or [[TMP6]] : i1
  // CHECK-NEXT: [[TMP8:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP9:%.+]] = comb.mux [[TMP7]], [[TMP8]], [[TMP3]] : i20
  // Check that the output mux is controlled by a conflict, and is between the random and intended read
  // CHECK-NEXT: hw.output [[TMP9]] : i20
  // CHECK-NEXT: }
  hw.module @ReadWriteConflict_BothEnabledOOB(in %data: i20, in %clock: !seq.clock,  out z: i20) {
  %enableRead = hw.constant true // Set to constant 1
  %enableWrite = hw.constant true // Set to constant 1

  %addr = hw.constant 13 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enableRead: <12 x 20>
  seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enableWrite: <12 x 20>

  hw.output %0 : i20
  }


//
// Read Write, Different Address
//-----

  // CHECK-LABEL: hw.module @ReadWriteConflict_NoConflict
  // CHECK-NEXT: [[ENABLEREAD:%.+]] = hw.constant true
  // CHECK-NEXT: [[ENABLEWRITE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDRREAD:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: [[ADDRWRITE:%.+]] = hw.constant 4 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDRREAD]]], clock %clock enable [[ENABLEREAD]] : <12 x 20>
  // READ OOB
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDRREAD]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // WRITE OOB
  // CHECK-NEXT: [[TMP0_W:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDRWRITE]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.parity [[TMP1_W]] : i1
  // CHECK-NEXT: [[NEW_ENABLEWRITE:%.+]] = comb.and [[TMP2_W]], [[ENABLEWRITE]] : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDRWRITE]]] = %data, clock %clock enable [[NEW_ENABLEWRITE]] : <12 x 20>
  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDRREAD]], [[ADDRWRITE]] : i4 
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLEREAD]], [[NEW_ENABLEWRITE]] : i1 
  // CHECK-NEXT: [[TMP6:%.+]] = comb.and [[TMP4]], [[TMP5]] : i1 
  // CHECK-NEXT: [[TMP7:%.+]] = comb.or [[TMP6]] : i1
  // CHECK-NEXT: [[TMP8:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP9:%.+]] = comb.mux [[TMP7]], [[TMP8]], [[TMP3]] : i20
  // Check that the output mux is controlled by a conflict, and is between the random and intended read
  // CHECK-NEXT: hw.output [[TMP9]] : i20
  // CHECK-NEXT: }

  hw.module @ReadWriteConflict_NoConflict(in %data: i20, in %clock: !seq.clock,  out z: i20) {
  %enableRead = hw.constant true // Set to constant 1
  %enableWrite = hw.constant true
  %addrread = hw.constant 6 : i4
  %addrwrite = hw.constant 4 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addrread], clock %clock enable %enableRead: <12 x 20>
  seq.firmem.write_port %mem[%addrwrite] = %data, clock %clock enable %enableWrite: <12 x 20>

  hw.output %0 : i20
  }



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~```
// Reading out of bounds

//
// READ OUT OF BOUNDS, DISABLED
//  WORKS
//------

  // CHECK-LABEL: hw.module @ReadOutOfBoundsDisabled
  // CHECK-NEXT: [[ENABLE:%.+]] = hw.constant false
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant -3 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLE]] : <12 x 20>
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // CHECK-NEXT: hw.output [[TMP3]] : i20
  // CHECK-NEXT: }
  hw.module @ReadOutOfBoundsDisabled(in %clock: !seq.clock, out z: i20) {
    %enable = hw.constant false
    %addr = hw.constant 13 : i4
    %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

    %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enable: <12 x 20>

    hw.output %0 : i20
  }
//

// READ OUT OF BOUNDS, ENABLED
// 
//------

  // CHECK-LABEL: hw.module @ReadOutOfBoundsEnabled
  // CHECK-NEXT: [[ENABLE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant -3 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLE]] : <12 x 20>
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // CHECK-NEXT: hw.output [[TMP3]] : i20
  // CHECK-NEXT: }
  hw.module @ReadOutOfBoundsEnabled(in %clock: !seq.clock,  out z: i20) {
    %enable = hw.constant true
    %addr = hw.constant 13 : i4
    %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

    %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enable: <12 x 20>

    hw.output %0 : i20
  }


// 

// READ IN BOUNDS, ENABLED
// WORKS
//------

  // CHECK-LABEL: hw.module @ReadInBoundsEnabled
  // CHECK-NEXT: [[ENABLE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLE]] : <12 x 20>
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
  // CHECK-NEXT: hw.output [[TMP3]] : i20
  // CHECK-NEXT: }
  hw.module @ReadInBoundsEnabled(in %clock: !seq.clock,  out z: i20) {
    %enable = hw.constant true
    %addr = hw.constant 6 : i4
    %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

    %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enable: <12 x 20>

    hw.output %0 : i20
  }

//
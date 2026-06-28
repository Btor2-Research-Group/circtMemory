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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Multiple Writes / Reads/ ReadWrites same address

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Read Read-Writes
// Conditions:


// 
// READ & READWRITE, BOTH ENABLED
// Q: More ports show in the report (checking for the read being out of bounds).

//------

// CHECK-LABEL: hw.module @Read_ReadWriteConflict_BothEnabled
// CHECK-NEXT : [[ENABLE:%.+]] = hw.constant true 
// CHECK-NEXT : [[MODE:%.+]] = hw.constant true
// CHECK-NEXT : [[ADDR:%.+]] = hw.constant 6 : i4
// CHECK-NEXT : %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
// CHECK-NEXT : [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLE]] : <12 x 20>
// CHECK-NEXT : [[WRITE:%.+]] = seq.firmem.read_write_port %mem[[[ADDR]]] = %data, clock %clock enable [[ENABLE]] : <12 x 20>
// CHECK-NEXT : [[TMP0:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
// CHECK-NEXT : [[TMP1:%.+]] = comb.and [[MODE]], [[ENABLE]] : i1
// CHECK-NEXT : [[TMP2:%.+]] = comb.and  [[ENABLE]], [[TMP1]] : i1 
// CHECK-NEXT : [[TMP3:%.+]] = comb.and [[TMP0]], [[TMP2]] : i1 
// CHECK-NEXT : [[TMP4:%.+]] = comb.or [[TMP3]], FALSE : i1
// CHECK-NEXT : [[TMP5:%.+]] = verif.symbolic_value : i20
// CHECK-NEXT : [[TMP6:%.+]] = comb.mux [[TMP4]], [[TMP5]], [[READ]] : i4
// Check that the output mux is controlled by a conflict, and is between the random and intended read
// CHECK-NEXT : hw.output [[TMP4]] : i20
// CHECK-NEXT : }
hw.module @Read_ReadWriteConflict_BothEnabled(in %data: i20, in %clock: !seq.clock, out z: i20) {
 %enable = hw.constant true // Set to constant 1
 %mode = hw.constant true // Set the ReadWrite to writing
 %addr = hw.constant 6 : i4
 %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

 %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enable: <12 x 20>
 %1 = seq.firmem.read_write_port %mem[%addr] = %data if %mode, clock %clock enable %enable: <12 x 20>

 hw.output %0 : i20
}






//
// ReadOp WriteOp Conflicts

// Both Enabled, One of each enabled, out of bounds.




// 
// READ WRITE, BOTH ENABLED
// -- WORKS

//------

// CHECK-LABEL: hw.module @ReadWriteConflict_BothEnabled
// Read Out of Bounds check
// CHECK-NEXT: [[ENABLE:%.+]] = hw.constant true
// CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
// CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
// CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLE]] : <12 x 20>
// CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
// CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
// CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
// CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20
// Read write
// CHECK-NEXT: [[WRITE:%.+]] = seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[ENABLE]] : <12 x 20>
// CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
// CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLE]], [[ENABLE]] : i1 
// CHECK-NEXT: [[TMP6:%.+]] = comb.and [[TMP4]], [[TMP5]] : i1 
// CHECK-NEXT: [[TMP7:%.+]] = comb.or [[TMP6]], FALSE : i1
// CHECK-NEXT: [[TMP8:%.+]] = verif.symbolic_value : i20
// CHECK-NEXT: [[TMP9:%.+]] = comb.mux [[TMP7]], [[TMP8]], [[TMP3]] : i4
// Check that the output mux is controlled by a conflict, and is between the random and intended read
// CHECK-NEXT: hw.output [[TMP9]] : i20
// CHECK-NEXT: }
hw.module @ReadWriteConflict_BothEnabled(in %data: i20, in %clock: !seq.clock,  out z: i20) {
 %enable = hw.constant true // Set to constant 1

 %addr = hw.constant 6 : i4
 %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

 %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enable: <12 x 20>
 seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enable: <12 x 20>

 hw.output %0 : i20
}






// Read write, Read disabled
// Works?
//------

// CHECK-LABEL: hw.module @ReadWriteConflict_ReadDisabled
// CHECK-NEXT : [[ENABLEREAD:%.+]] = hw.constant false 
// CHECK-NEXT : [[ENABLEWRITE:%.+]] = hw.constant true 
// CHECK-NEXT : [[ADDR:%.+]] = hw.constant 6 : i4
// CHECK-NEXT : [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLEREAD]] : <12 x 20>
// CHECK-NEXT : [[WRITE:%.+]] = seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[ENABLEWRITE]] : <12 x 20>
// CHECK-NEXT : [[TMP0:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
// CHECK-NEXT : [[TMP1:%.+]] = comb.and  [[ENABLEREAD]], [[ENABLEWRITE]] : i1 
// CHECK-NEXT : [[TMP2:%.+]] = comb.and [[TMP0]], [[TMP1]] : i1 
// CHECK-NEXT : [[TMP3:%.+]] = comb.or [[TMP2]], FALSE : i1
// CHECK-NEXT : [[TMP4:%.+]] = verif.symbolic_value : i20
// CHECK-NEXT : [[TMP5:%.+]] = comb.mux [[TMP3]], [[TMP4]], [[READ]] : i4
// Check that the output mux is controlled by a conflict, and is between the random and intended read
// CHECK-NEXT : hw.output [[TMP3]] : i20
// CHECK-NEXT : }
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
// Works?
//-----
// CHECK-LABEL: hw.module @ReadWriteConflict_WriteDisabled
// CHECK-NEXT : [[ENABLEREAD:%.+]] = hw.constant true 
// CHECK-NEXT : [[ENABLEWRITE:%.+]] = hw.constant false 
// CHECK-NEXT : [[ADDR:%.+]] = hw.constant 6 : i4
// CHECK-NEXT : [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLEREAD]] : <12 x 20>
// CHECK-NEXT : [[WRITE:%.+]] = seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[ENABLEWRITE]] : <12 x 20>
// CHECK-NEXT : [[TMP0:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
// CHECK-NEXT : [[TMP1:%.+]] = comb.and  [[ENABLEREAD]], [[ENABLEWRITE]] : i1 
// CHECK-NEXT : [[TMP2:%.+]] = comb.and [[TMP0]], [[TMP1]] : i1 
// CHECK-NEXT : [[TMP3:%.+]] = comb.or [[TMP2]], FALSE : i1
// CHECK-NEXT : [[TMP4:%.+]] = verif.symbolic_value : i20
// CHECK-NEXT : [[TMP5:%.+]] = comb.mux [[TMP3]], [[TMP4]], [[READ]] : i4
// Check that the output mux is controlled by a conflict, and is between the random and intended read
// CHECK-NEXT : hw.output [[TMP3]] : i20
// CHECK-NEXT : }
hw.module @ReadWriteConflict_WriteDisabled(in %data: i20, in %clock: !seq.clock,  out z: i20) {
 %enableRead = hw.constant true // Set to constant 1
 %enableWrite = hw.constant false
 %addr = hw.constant 6 : i4
 %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

 %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enableRead: <12 x 20>
 seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enableWrite: <12 x 20>

 hw.output %0 : i20
}

// TODO:
// Read write, both enabled to the same out of bounds address
//------

// CHECK-LABEL : hw.module @ReadWriteConflict_BothEnabledOOB
// CHECK-NEXT : [[ENABLE:%.+]] = hw.constant true 
// CHECK-NEXT : [[ADDR:%.+]] = hw.constant -3 : i4
// CHECK-NEXT : [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLE]] : <12 x 20>
// CHECK-NEXT : [[WRITE:%.+]] = seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[ENABLE]] : <12 x 20>
// CHECK-NEXT : [[TMP0:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
// CHECK-NEXT : [[TMP1:%.+]] = comb.and  [[ENABLE]], [[ENABLE]] : i1 
// CHECK-NEXT : [[TMP2:%.+]] = comb.and [[TMP0]], [[TMP1]] : i1 
// CHECK-NEXT : [[TMP3:%.+]] = comb.or [[TMP2]], FALSE : i1
// CHECK-NEXT : [[TMP4:%.+]] = verif.symbolic_value : i20
// CHECK-NEXT : [[TMP5:%.+]] = comb.mux [[TMP3]], [[TMP4]], [[READ]] : i4
// Check that the output mux is controlled by a conflict, and is between the random and intended read
// CHECK-NEXT : hw.output [[TMP3]] : i20
// CHECK-NEXT : }
hw.module @ReadWriteConflict_BothEnabledOOB(in %data: i20, in %clock: !seq.clock,  out z: i20) {
 %enable = hw.constant true // Set to constant 1

 %addr = hw.constant 13 : i4
 %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

 %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enable: <12 x 20>
 seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enable: <12 x 20>

 hw.output %0 : i20
}




//
// Reading out of bounds



//
// READ OUT OF BOUNDS, DISABLED
// CHECKED - WORKS

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
// CHECKED - WORKS

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
// CHECKED - WORKS

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



// NEED: READ WRITE AS A READ OUT OF BOUNDS ENABLED/DISABLED
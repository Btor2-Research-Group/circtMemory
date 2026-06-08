// RUN: circt-opt --undefined-memory-behavior --split-input-file %s | FileCheck %s

// ton of binaries
// look for circt-opt

// circt-opt
// https://github.com/chipsalliance/firrtl/blob/1.6.x/src/test/scala/firrtl/backends/experimental/smt/random/UndefinedMemoryBehaviorSpec.scala


// //CHECK-LABEL: hw.module @ReadWriteConflict // do i check isOutOfBounds, muxForOOB as well?
// //CHECK: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
// //CHECK: %0 =  comb.icmp eq %addr, %addr : i4 // same address
// //CHECK: %1 = comb.and %enable, %enable : i1  // read and write enabled
// //CHECK: %2 = comb.and %enable, %enable : i1                 // is collision




// The addresses are equal
// //CHECK-LABEL: hw.module @ReadWriteConflict
// //CHECK-NEXT: [[TMP0:%.+]] = comb.icmp eq %addr, %addr : i4 // Same address
// //CHECK-NEXT: [[TMP1:%.+]] = comb.and  %enable, %enable :i1 // both enabled
// //CHECK-NEXT: [[TMP2:%.+]] = comb.and [[TMP0]], [[TMP1]] : i1 // Is Collision
// In the code we make an OR operator: is this needed?
// //CHECK-NEXT: [[TMP3:%.+]] = SymbolicValue.op// Random Value
// //CHECK-NEXT: []TMP4:%.+]] = comb.muxOp [[TMP2]], ?? ??

//hw.module @ReadWriteConflict(in %addr: i4, in %clock: !seq.clock, in %enable: i1, out z: i20) {
//  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

//  %0 = seq.firmem.read_port %mem[%address], clock %clock enable %enable: <12 x 20>
//  seq.firmem.write_port %mem[%address] = %data, clock %clock enable %enable: <12 x 20>

//  hw.output %0 : i20
//}



// //CHECK-LABEL: hw.module @ManyAssertsAndAssumes
// //CHECK-NEXT:   %c1_i42 = hw.constant 1 : i42
// //CHECK-NEXT:   [[TMP0:%.+]] = comb.shl %a, %c1_i42 : i42
// //CHECK-NEXT:   %c2_i42 = hw.constant 2 : i42
// //CHECK-NEXT:   [[TMP1:%.+]] = comb.icmp ult %a, %c2_i42 : i42
// //CHECK-NEXT:   %c0_i42 = hw.constant 0 : i42
// //CHECK-NEXT:   [[TMP2:%.+]] = comb.icmp uge %a, %c0_i42 : i42
// //CHECK-NEXT:   [[REQ:%.+]] = comb.and [[TMP1]], [[TMP2]] : i1
// //CHECK-NEXT:   verif.assume [[REQ]] : i1
// //CHECK-NEXT:   [[TMP3:%.+]] = comb.mul %a, %c2_i42 : i42
// //CHECK-NEXT:   [[TMP4:%.+]] = comb.icmp eq [[TMP0]], [[TMP3]] : i42
// //CHECK-NEXT:   [[TMP5:%.+]] = comb.add %a, %a : i42
// //CHECK-NEXT:   [[TMP6:%.+]] = comb.icmp eq [[TMP0]], [[TMP5]] : i42
// //CHECK-NEXT:   [[ENS:%.+]] = comb.and [[TMP4]], [[TMP6]] : i1
// //CHECK-NEXT:   verif.assert [[ENS]] : i1
// //CHECK-NEXT:   hw.output [[TMP0]] : i42
// //CHECK-NEXT: }




// Check that out of bounds variable is active
// Check that the data is random

// CHECK-LABEL: hw.module @ReadWriteConflict // do i check isOutOfBounds, muxForOOB as well?
// CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20> // Check that memory exists
// CHECK-NEXT: [[TMP0:%.+]]  = hw.constant 12 : i64 // Depth: Needs to exist, because depth may not be constant across systems
// CHECK-NEXT: [[TMP1:%.+]]  = comp.icmp gt  %addr, [[TMP0]] : i4 // Check that the address is no longer within the depth (m_r_oob in scala)
// CHECK-NEXT: [[TMP2:%.+]]  = SymbolicValue.op  // RANDOM DATA is generated
// CHECK-NEXT: [[TMP3:%.+]]  = comb.muxOp [[TMP1]], [[TMP2]], {{.*}}: i4 // RANDOM data is chosen. Do we need this? How do we choose the "old" data - is there a vague way to to it?
hw.module @ReadOutOfBounds(in %addr: i4, in %clock: !seq.clock, in %enable: i1, out z: i20) {
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  // How do I ensure that this is our of bounds? Is this accurate?
  %0 = seq.firmem.read_port %mem[13 * %address], clock %clock enable %enable: <12 x 20>

  hw.output %0 : i20
}


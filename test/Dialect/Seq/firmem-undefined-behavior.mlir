// RUN: circt-opt --undefined-memory-behavior --split-input-file %s | FileCheck %s

// ton of binaries
// look for circt-opt

// circt-opt
// 

// CHECK-LABEL: hw.module @ReadWriteConflict
// CHECK: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
// CHECK: %0
// CHECK: %1 = comb.and %enable, %enable : i1
// CHECK: %2 = comb.and 

// Need case where read out of bounds

hw.module @ReadWriteConflict(in %addr: i4, in %clock: !seq.clock, in %enable: i1, out z: i20) {
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%address], clock %clock enable %enable: <12 x 20>
  seq.firmem.write_port %mem[%address] = %data, clock %clock enable %enable: <12 x 20>

  hw.output %0 : i20
}


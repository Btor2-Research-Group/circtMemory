// RUN: circt-opt --undefined-memory-behavior --split-input-file %s | FileCheck %s
// circt-opt --undefined-memory-behavior tests/Dialect/Seq/firmem-undefined-behavior.mlir

// TODO: CHECK LOGICAL EQUIVALENCE WITH SCALA using a script - bash or python that has an example design and proves they are the same.
// Integration test. Covers a ton of conflicts. Run the design through the scala-firrtl, get btor from each then compare the two through midr circuit

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Multiple Writes / Reads/ ReadWrites same address




//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Write Conflicts

// Write OOB
//-----

  //CHECK-LABEL: hw.module @WriteConflict_OOB
  // Read Out of Bounds check
  // CHECK-NEXT: [[ENABLE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // OOB Check
  // WRITE OOB
  // CHECK-NEXT: [[TMP0_W:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.xor [[TMP1_W]], [[TRUE]] : i1
  // CHECK-NEXT: [[TMP3_W:%.+]] = comb.and [[TMP2_W]], [[ENABLE]] : i1
  // CHECK-NEXT: verif.assert [[TMP3_W]] label "write_enable" : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[ENABLE]] : <12 x 20>
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
//

// Write-Write Conflict
// -----
  //CHECK-LABEL: hw.module @Write_WriteConflict
  // Read Out of Bounds check
  // CHECK-NEXT: [[ENABLE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // OOB Check
  // FIRST WRITE
  // CHECK-NEXT: [[TMP0_W:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.xor [[TMP1_W]], [[TRUE]] : i1
  // CHECK-NEXT: [[TMP3_W:%.+]] = comb.and [[TMP2_W]], [[ENABLE]] : i1
  // CHECK-NEXT: verif.assert [[TMP3_W]] label "write_enable" : i1
  // Write-write conflict, first write
  // CHECK-NEXT: [[SAME_ADDR1:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4
  // CHECK-NEXT: [[BOTH_ENABLED1:%.+]] = comb.and [[ENABLE]], [[ENABLE]] : i1
  // CHECK-NEXT: [[SAME_ADDR_ENABLED:%.+]] = comb.and [[SAME_ADDR1]], [[BOTH_ENABLED1]] : i1
  // Look for collisions
  // CHECK-NEXT: [[COLLISION1:%.+]] = comb.or [[SAME_ADDR_ENABLED]] : i1
  // CHECK-NEXT: [[RANDOM_DATA:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[RAND_DATA_MUX:%.+]] = comb.mux [[COLLISION1]], [[RANDOM_DATA]], %data : i20
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = [[RAND_DATA_MUX]], clock %clock enable [[ENABLE]] : <12 x 20>
  // SECOND WRITE
  // CHECK-NEXT: [[TMP0_W2:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TRUE2:%.+]] = hw.constant true
  // CHECK-NEXT: [[TMP1_W2:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W2]] : i4
  // CHECK-NEXT: [[TMP2_W2:%.+]] = comb.xor [[TMP1_W2]], [[TRUE2]] : i1
  // CHECK-NEXT: [[TMP3_W2:%.+]] = comb.and [[TMP2_W2]], [[ENABLE]] : i1
  // CHECK-NEXT: verif.assert [[TMP3_W2]] label "write_enable" : i1
  // Write-write conflict, first write
  // CHECK-NEXT: [[SAME_ADDR2:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4
  // CHECK-NEXT: [[BOTH_ENABLED2:%.+]] = comb.and [[ENABLE]], [[ENABLE]] : i1
  // CHECK-NEXT: [[SAME_ADDR_ENABLED2:%.+]] = comb.and [[SAME_ADDR2]], [[BOTH_ENABLED2]] : i1
  // Look for collisions
  // CHECK-NEXT: [[COLLISION2:%.+]] = comb.or [[SAME_ADDR_ENABLED2]] : i1
  // CHECK-NEXT: [[RANDOM_DATA2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[RAND_DATA_MUX2:%.+]] = comb.mux [[COLLISION2]], [[RANDOM_DATA2]], %data : i20
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = [[RAND_DATA_MUX2]], clock %clock enable [[ENABLE]] : <12 x 20>
  // CHECK-NEXT: hw.output [[ADDR]] : i4
  // CHECK-NEXT: }
    hw.module @Write_WriteConflict(in %data: i20, in %clock: !seq.clock,  out z: i4) {
    %enable = hw.constant true // Set to constant 1
    %addr = hw.constant 6 : i4
    %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

    seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enable: <12 x 20>
    seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enable: <12 x 20>

    // Output something.
      hw.output %addr : i4
    }
//






//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Read, Read Write Conflict
// Types: read/write collisions, but for read write.
// Both Enabled, Both disabled, one enabled one not, Diff address

// Read, Read Write Both Enabled
// 
//------

  // CHECK-LABEL: hw.module @Read_ReadWriteConflict_BothEnabled
  // CHECK-NEXT: [[ENABLERD:%.+]] = hw.constant true
  // CHECK-NEXT: [[ENABLERW:%.+]] = hw.constant true
  // CHECK-NEXT: [[MODE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDRRD:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: [[ADDRRW:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDRRD]]], clock %clock enable [[ENABLERD]] : <12 x 20>
  // Read OOB Check
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDRRD]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20

  // Read Write OOB Write Check
    // Read Out of Bounds check
    // CHECK-NEXT: [[WRITEENABLED:%.+]] = comb.and [[MODE]], [[ENABLERW]]
    // Get Read Enable
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[READMODE:%.+]] = comb.xor [[MODE]], [[TRUE]]
    // CHECK-NEXT: [[READ_ENABLED:%.+]] = comb.and [[READMODE]], [[ENABLERW]]
    // WRITE OOB
    // CHECK-NEXT: [[DEPTH:%.+]] = hw.constant -4 : i4
    // CHECK-NEXT: [[OOB:%.+]] = comb.icmp uge [[ADDRRW]], [[DEPTH]] : i4
    // CHECK-NEXT: [[NOTOOB:%.+]] = comb.xor [[OOB]], [[TRUE]]
    // CHECK-NEXT: [[TMP4_W:%.+]] = comb.and [[NOTOOB]], [[WRITEENABLED]] : i1
    // CHECK-NEXT: verif.assert [[TMP4_W]] label "write_enable" : i1

    // READ OOB
    // CHECK-NEXT: [[READ_RW1:%.+]] = seq.firmem.read_write_port %mem[[[ADDRRW]]] = %data if [[MODE]], clock %clock enable [[ENABLERW]] : <12 x 20>
    // CHECK-NEXT: [[RAND:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[READOOB:%.+]] = comb.and [[OOB]], [[READ_ENABLED]] : i1
    // CHECK-NEXT: [[READ_MUX_RW:%.+]] = comb.mux [[READOOB]], [[RAND]], [[READ_RW1]]

  // Read - Read Write Conflict Check
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDRRD]], [[ADDRRW]] : i4 
  // Is the Read-Write Enabled/In write mode?
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and [[MODE]], [[ENABLERW]] : i1 
  // Are both Enabled?
  // CHECK-NEXT: [[TMP6:%.+]] = comb.and [[ENABLERD]], [[TMP5]] : i1 
  // Are they the same addr & enabled?
  // CHECK-NEXT: [[TMP7:%.+]] = comb.and [[TMP4]], [[TMP6]] : i1 
  // CHECK-NEXT: [[TMP8:%.+]] = comb.or [[TMP7]] : i1
  // CHECK-NEXT: [[TMP9:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP10:%.+]] = comb.mux [[TMP8]], [[TMP9]], [[TMP3]] : i20

  // CHECK-NEXT: hw.output [[TMP10]] : i20
  // CHECK-NEXT: }
  hw.module @Read_ReadWriteConflict_BothEnabled(in %data: i20, in %clock: !seq.clock, out z: i20) {
  %enableRD = hw.constant true // Set to constant 1
  %enableRW = hw.constant true // Set to constant 1

  %mode = hw.constant true // Set the ReadWrite to writing
  %addrRD = hw.constant 6 : i4
  %addrRW = hw.constant 6 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addrRD], clock %clock enable %enableRD: <12 x 20>
  %1 = seq.firmem.read_write_port %mem[%addrRW] = %data if %mode, clock %clock enable %enableRW: <12 x 20>

  hw.output %0 : i20
  }
//


// Read, Read Write, Read Enabled
// 
//------

  // CHECK-LABEL: hw.module @Read_ReadWriteConflict_ReadEnabled
  // CHECK-NEXT: [[ENABLERD:%.+]] = hw.constant true
  // CHECK-NEXT: [[ENABLERW:%.+]] = hw.constant false
  // CHECK-NEXT: [[MODE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDRRD:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: [[ADDRRW:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDRRD]]], clock %clock enable [[ENABLERD]] : <12 x 20>
  // Read OOB Check
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDRRD]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20

  // Read Write OOB Write Check
    // Read Out of Bounds check
    // CHECK-NEXT: [[WRITEENABLED:%.+]] = comb.and [[MODE]], [[ENABLERW]]
    // Get Read Enable
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[READMODE:%.+]] = comb.xor [[MODE]], [[TRUE]]
    // CHECK-NEXT: [[READ_ENABLED:%.+]] = comb.and [[READMODE]], [[ENABLERW]]
    // WRITE OOB
    // CHECK-NEXT: [[DEPTH:%.+]] = hw.constant -4 : i4
    // CHECK-NEXT: [[OOB:%.+]] = comb.icmp uge [[ADDRRW]], [[DEPTH]] : i4
    // CHECK-NEXT: [[NOTOOB:%.+]] = comb.xor [[OOB]], [[TRUE]]
    // CHECK-NEXT: [[TMP4_W:%.+]] = comb.and [[NOTOOB]], [[WRITEENABLED]] : i1
    // CHECK-NEXT: verif.assert [[TMP4_W]] label "write_enable" : i1

    // READ OOB
    // CHECK-NEXT: [[READ_RW1:%.+]] = seq.firmem.read_write_port %mem[[[ADDRRW]]] = %data if [[MODE]], clock %clock enable [[ENABLERW]] : <12 x 20>
    // CHECK-NEXT: [[RAND:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[READOOB:%.+]] = comb.and [[OOB]], [[READ_ENABLED]] : i1
    // CHECK-NEXT: [[READ_MUX_RW:%.+]] = comb.mux [[READOOB]], [[RAND]], [[READ_RW1]]

  // Read - Read Write Conflict Check
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDRRD]], [[ADDRRW]] : i4 
  // Is the Read-Write Enabled/In write mode?
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and [[MODE]], [[ENABLERW]] : i1 
  // Are both Enabled?
  // CHECK-NEXT: [[TMP6:%.+]] = comb.and [[ENABLERD]], [[TMP5]] : i1 
  // Are they the same addr & enabled?
  // CHECK-NEXT: [[TMP7:%.+]] = comb.and [[TMP4]], [[TMP6]] : i1 
  // CHECK-NEXT: [[TMP8:%.+]] = comb.or [[TMP7]] : i1
  // CHECK-NEXT: [[TMP9:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP10:%.+]] = comb.mux [[TMP8]], [[TMP9]], [[TMP3]] : i20

  // CHECK-NEXT: hw.output [[TMP10]] : i20
  // CHECK-NEXT: }
  hw.module @Read_ReadWriteConflict_ReadEnabled(in %data: i20, in %clock: !seq.clock, out z: i20) {
  %enableREAD = hw.constant true // Set to constant 1
  %enableRW = hw.constant false
  %mode = hw.constant true // Set the ReadWrite to writing
  %addr = hw.constant 6 : i4
  %addrRW = hw.constant 6 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enableREAD: <12 x 20>
  %1 = seq.firmem.read_write_port %mem[%addrRW] = %data if %mode, clock %clock enable %enableRW: <12 x 20>

  hw.output %0 : i20
  }
//






// Read, Read Write, Both Disabled
//
//------

  // CHECK-LABEL: hw.module @Read_ReadWriteConflict_Disabled
  // CHECK-NEXT: [[ENABLERD:%.+]] = hw.constant false
  // CHECK-NEXT: [[ENABLERW:%.+]] = hw.constant false
  // CHECK-NEXT: [[MODE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDRRD:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: [[ADDRRW:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDRRD]]], clock %clock enable [[ENABLERD]] : <12 x 20>
  // Read OOB Check
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDRRD]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20

  // Read Write OOB Write Check
    // Read Out of Bounds check
    // CHECK-NEXT: [[WRITEENABLED:%.+]] = comb.and [[MODE]], [[ENABLERW]]
    // Get Read Enable
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[READMODE:%.+]] = comb.xor [[MODE]], [[TRUE]]
    // CHECK-NEXT: [[READ_ENABLED:%.+]] = comb.and [[READMODE]], [[ENABLERW]]
    // WRITE OOB
    // CHECK-NEXT: [[DEPTH:%.+]] = hw.constant -4 : i4
    // CHECK-NEXT: [[OOB:%.+]] = comb.icmp uge [[ADDRRW]], [[DEPTH]] : i4
    // CHECK-NEXT: [[NOTOOB:%.+]] = comb.xor [[OOB]], [[TRUE]]
    // CHECK-NEXT: [[TMP4_W:%.+]] = comb.and [[NOTOOB]], [[WRITEENABLED]] : i1
    // CHECK-NEXT: verif.assert [[TMP4_W]] label "write_enable" : i1

    // READ OOB
    // CHECK-NEXT: [[READ_RW1:%.+]] = seq.firmem.read_write_port %mem[[[ADDRRW]]] = %data if [[MODE]], clock %clock enable [[ENABLERW]] : <12 x 20>
    // CHECK-NEXT: [[RAND:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[READOOB:%.+]] = comb.and [[OOB]], [[READ_ENABLED]] : i1
    // CHECK-NEXT: [[READ_MUX_RW:%.+]] = comb.mux [[READOOB]], [[RAND]], [[READ_RW1]]

  // Read - Read Write Conflict Check
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDRRD]], [[ADDRRW]] : i4 
  // Is the Read-Write Enabled/In write mode?
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and [[MODE]], [[ENABLERW]] : i1 
  // Are both Enabled?
  // CHECK-NEXT: [[TMP6:%.+]] = comb.and [[ENABLERD]], [[TMP5]] : i1 
  // Are they the same addr & enabled?
  // CHECK-NEXT: [[TMP7:%.+]] = comb.and [[TMP4]], [[TMP6]] : i1 
  // CHECK-NEXT: [[TMP8:%.+]] = comb.or [[TMP7]] : i1
  // CHECK-NEXT: [[TMP9:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP10:%.+]] = comb.mux [[TMP8]], [[TMP9]], [[TMP3]] : i20

  // CHECK-NEXT: hw.output [[TMP10]] : i20
  // CHECK-NEXT: }
  hw.module @Read_ReadWriteConflict_Disabled(in %data: i20, in %clock: !seq.clock, out z: i20) {
  %enableREAD = hw.constant false // Set to constant 1
  %enableRW = hw.constant false
  %mode = hw.constant true // Set the ReadWrite to writing
  %addr = hw.constant 6 : i4
  %addrRW = hw.constant 6 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enableREAD: <12 x 20>
  %1 = seq.firmem.read_write_port %mem[%addrRW] = %data if %mode, clock %clock enable %enableRW: <12 x 20>

  hw.output %0 : i20
  }
//


// Read, Read Write - Different Address
// 
//------

  // CHECK-LABEL: hw.module @Read_ReadWriteConflict_DiffAddr
  // CHECK-NEXT: [[ENABLERD:%.+]] = hw.constant true
  // CHECK-NEXT: [[ENABLERW:%.+]] = hw.constant true
  // CHECK-NEXT: [[MODE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDRRD:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: [[ADDRRW:%.+]] = hw.constant 4 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDRRD]]], clock %clock enable [[ENABLERD]] : <12 x 20>
  // Read OOB Check
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDRRD]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20

  // Read Write OOB Write Check
    // Read Out of Bounds check
    // CHECK-NEXT: [[WRITEENABLED:%.+]] = comb.and [[MODE]], [[ENABLERW]]
    // Get Read Enable
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[READMODE:%.+]] = comb.xor [[MODE]], [[TRUE]]
    // CHECK-NEXT: [[READ_ENABLED:%.+]] = comb.and [[READMODE]], [[ENABLERW]]
    // WRITE OOB
    // CHECK-NEXT: [[DEPTH:%.+]] = hw.constant -4 : i4
    // CHECK-NEXT: [[OOB:%.+]] = comb.icmp uge [[ADDRRW]], [[DEPTH]] : i4
    // CHECK-NEXT: [[NOTOOB:%.+]] = comb.xor [[OOB]], [[TRUE]]
    // CHECK-NEXT: [[TMP4_W:%.+]] = comb.and [[NOTOOB]], [[WRITEENABLED]] : i1
    // CHECK-NEXT: verif.assert [[TMP4_W]] label "write_enable" : i1

    // READ OOB
    // CHECK-NEXT: [[READ_RW1:%.+]] = seq.firmem.read_write_port %mem[[[ADDRRW]]] = %data if [[MODE]], clock %clock enable [[ENABLERW]] : <12 x 20>
    // CHECK-NEXT: [[RAND:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[READOOB:%.+]] = comb.and [[OOB]], [[READ_ENABLED]] : i1
    // CHECK-NEXT: [[READ_MUX_RW:%.+]] = comb.mux [[READOOB]], [[RAND]], [[READ_RW1]]

  // Read - Read Write Conflict Check
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDRRD]], [[ADDRRW]] : i4 
  // Is the Read-Write Enabled/In write mode?
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and [[MODE]], [[ENABLERW]] : i1 
  // Are both Enabled?
  // CHECK-NEXT: [[TMP6:%.+]] = comb.and [[ENABLERD]], [[TMP5]] : i1 
  // Are they the same addr & enabled?
  // CHECK-NEXT: [[TMP7:%.+]] = comb.and [[TMP4]], [[TMP6]] : i1 
  // CHECK-NEXT: [[TMP8:%.+]] = comb.or [[TMP7]] : i1
  // CHECK-NEXT: [[TMP9:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP10:%.+]] = comb.mux [[TMP8]], [[TMP9]], [[TMP3]] : i20

  // CHECK-NEXT: hw.output [[TMP10]] : i20
  // CHECK-NEXT: }
  hw.module @Read_ReadWriteConflict_DiffAddr(in %data: i20, in %clock: !seq.clock, out z: i20) {
  %enable = hw.constant true // Set to constant 1
  %enableRW = hw.constant true
  %mode = hw.constant true // Set the ReadWrite to writing
  %addrREAD = hw.constant 6 : i4
  %addrRW = hw.constant 4 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addrREAD], clock %clock enable %enable: <12 x 20>
  %1 = seq.firmem.read_write_port %mem[%addrRW] = %data if %mode, clock %clock enable %enableRW: <12 x 20>

  hw.output %0 : i20
  }
//





//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// READ WRITE CONFLICTS
// Both Enabled, One of each enabled, out of bounds.

// 
// READ WRITE, BOTH ENABLED
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
  // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.xor [[TMP1_W]], [[TRUE]] : i1
  // CHECK-NEXT: [[TMP3_W:%.+]] = comb.and [[TMP2_W]], [[ENABLEWRITE]] : i1
  // CHECK-NEXT: verif.assert [[TMP3_W]] label "write_enable" : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[ENABLEWRITE]] : <12 x 20>

  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLEREAD]], [[ENABLEWRITE]] : i1 
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

// 
// READ WRITE, Write does not have an enable
//------

  // CHECK-LABEL: hw.module @ReadWriteConflict_WriteNoEnable
  // Read Out of Bounds check
  // CHECK-NEXT: [[ENABLEREAD:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_port %mem[[[ADDR]]], clock %clock enable [[ENABLEREAD]] : <12 x 20>

  // READ OOB
  // CHECK-NEXT: [[TMP0:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TMP1:%.+]] = comb.icmp uge [[ADDR]], [[TMP0]] : i4
  // CHECK-NEXT: [[TMP2:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP3:%.+]] = comb.mux [[TMP1]], [[TMP2]], [[READ]] : i20

  // WRITE OOB
  // CHECK-NEXT: [[ENABLEWRITE:%.+]] = hw.constant true
  // CHECK-NEXT: [[TMP0_W:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.xor [[TMP1_W]], [[TRUE]] : i1
  // CHECK-NEXT: [[TMP3_W:%.+]] = comb.and [[TMP2_W]], [[ENABLEWRITE]] : i1
  // CHECK-NEXT: verif.assert [[TMP3_W]] label "write_enable" : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock : <12 x 20>

  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // CHECK-NEXT: [[ENABLEWRITE2:%.+]] = hw.constant true
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLEREAD]], [[ENABLEWRITE2]] : i1 
  // CHECK-NEXT: [[TMP6:%.+]] = comb.and [[TMP4]], [[TMP5]] : i1 
  // CHECK-NEXT: [[TMP7:%.+]] = comb.or [[TMP6]] : i1
  // CHECK-NEXT: [[TMP8:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[TMP9:%.+]] = comb.mux [[TMP7]], [[TMP8]], [[TMP3]] : i20
  // Check that the output mux is controlled by a conflict, and is between the random and intended read
  // CHECK-NEXT: hw.output [[TMP9]] : i20
  // CHECK-NEXT: }
  hw.module @ReadWriteConflict_WriteNoEnable(in %data: i20, in %clock: !seq.clock,  out z: i20) {
  %enableRead = hw.constant true // Set to constant 1
  // %enableWrite = hw.constant true // Set to constant 1
  %addr = hw.constant 6 : i4
  %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

  %0 = seq.firmem.read_port %mem[%addr], clock %clock enable %enableRead: <12 x 20>
  seq.firmem.write_port %mem[%addr] = %data, clock %clock : <12 x 20>

  hw.output %0 : i20
  }
//


// Read write, Read disabled
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
  // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.xor [[TMP1_W]], [[TRUE]] : i1
  // CHECK-NEXT: [[TMP3_W:%.+]] = comb.and [[TMP2_W]], [[ENABLEWRITE]] : i1
  // CHECK-NEXT: verif.assert [[TMP3_W]] label "write_enable" : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[ENABLEWRITE]] : <12 x 20>
  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLEREAD]], [[ENABLEWRITE]] : i1 
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

//

// Read write, write disabled
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
  // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.xor [[TMP1_W]], [[TRUE]] : i1
  // CHECK-NEXT: [[TMP3_W:%.+]] = comb.and [[TMP2_W]], [[ENABLEWRITE]] : i1
  // CHECK-NEXT: verif.assert [[TMP3_W]] label "write_enable" : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[ENABLEWRITE]] : <12 x 20>
  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLEREAD]], [[ENABLEWRITE]] : i1 
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
  // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.xor [[TMP1_W]], [[TRUE]] : i1
  // CHECK-NEXT: [[TMP3_W:%.+]] = comb.and [[TMP2_W]], [[ENABLEWRITE]] : i1
  // CHECK-NEXT: verif.assert [[TMP3_W]] label "write_enable" : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = %data, clock %clock enable [[ENABLEWRITE]] : <12 x 20>
  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4 
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLEREAD]], [[ENABLEWRITE]] : i1 
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
  // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDRWRITE]], [[TMP0_W]] : i4
  // CHECK-NEXT: [[TMP2_W:%.+]] = comb.xor [[TMP1_W]], [[TRUE]] : i1
  // CHECK-NEXT: [[TMP3_W:%.+]] = comb.and [[TMP2_W]], [[ENABLEWRITE]] : i1
  // CHECK-NEXT: verif.assert [[TMP3_W]] label "write_enable" : i1
  // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDRWRITE]]] = %data, clock %clock enable [[ENABLEWRITE]] : <12 x 20>
  // Read write
  // CHECK-NEXT: [[TMP4:%.+]] = comb.icmp eq [[ADDRREAD]], [[ADDRWRITE]] : i4 
  // CHECK-NEXT: [[TMP5:%.+]] = comb.and  [[ENABLEREAD]], [[ENABLEWRITE]] : i1 
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

//


//Test Case
  hw.module @ReadWriteSmem(in %clock : !seq.clock, in %reset : i1, in %io_enable : i1, in %io_write : i1, in %io_addr : i10, in %io_dataIn : i32, out io_dataOut : i32) {
    %mem = seq.firmem 1, 1, undefined, port_order : <1024 x 32>
    seq.firmem.write_port %mem[%io_addr] = %io_dataIn, clock %clock : <1024 x 32>
    %0 = seq.firmem.read_port %mem[%io_addr], clock %clock enable %io_enable {sv.namehint = "io_dataOut"} : <1024 x 32>
    hw.output %0 : i32
  }

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~```
// Reading out of bounds

//
// READ OUT OF BOUNDS, DISABLED
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


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~```
// Read Write Out Of Bounds Check


// Read Write OOB
//-----
  //CHECK-LABEL: hw.module @ReadWrite_OOB
  // Read Out of Bounds check
  // CHECK-NEXT: [[ENABLE:%.+]] = hw.constant true
  // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
  // CHECK-NEXT: [[MODE:%.+]] = hw.constant true
  // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
  // CHECK-NEXT: [[WRITEENABLED:%.+]] = comb.and [[MODE]], [[ENABLE]]
  // Get Read Enable
  // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
  // CHECK-NEXT: [[READMODE:%.+]] = comb.xor [[MODE]], [[TRUE]]
  // CHECK-NEXT: [[READ_ENABLED:%.+]] = comb.and [[READMODE]], [[ENABLE]]
  // WRITE OOB
  // CHECK-NEXT: [[DEPTH:%.+]] = hw.constant -4 : i4
  // CHECK-NEXT: [[OOB:%.+]] = comb.icmp uge [[ADDR]], [[DEPTH]] : i4
  // CHECK-NEXT: [[NOTOOB:%.+]] = comb.xor [[OOB]], [[TRUE]]
  // CHECK-NEXT: [[TMP4_W:%.+]] = comb.and [[NOTOOB]], [[WRITEENABLED]] : i1
  // CHECK-NEXT: verif.assert [[TMP4_W]] label "write_enable" : i1
  // READ OOB
  // CHECK-NEXT: [[READ:%.+]] = seq.firmem.read_write_port %mem[[[ADDR]]] = %data if [[MODE]], clock %clock enable [[ENABLE]] : <12 x 20>
  // CHECK-NEXT: [[RAND:%.+]] = verif.symbolic_value : i20
  // CHECK-NEXT: [[READOOB:%.+]] = comb.and [[OOB]], [[READ_ENABLED]] : i1
  // CHECK-NEXT: [[READ_MUX:%.+]] = comb.mux [[READOOB]], [[RAND]], [[READ]]
  // CHECK-NEXT: hw.output [[READ_MUX]] : i20
  // CHECK-NEXT: }
    hw.module @ReadWrite_OOB(in %data: i20, in %clock: !seq.clock,  out z: i20) {
    %enable = hw.constant true // Set to constant 1
    %addr = hw.constant 6 : i4
    %mode = hw.constant true
    %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
    

    %0 = seq.firmem.read_write_port %mem[%addr] = %data if %mode, clock %clock enable %enable: <12 x 20>

    // Does a write have an output?
      hw.output %0: i20
    }
//



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Read-Write, Write Conflict
// Types: Both write. Both enabled, one of each enabled, different address, out of bounds.
// Read-write is a read: all read-write collisions with a read-write/write pair.


  // Read-Write, Write Conflict
  //-----

    //CHECK-LABEL: hw.module @ReadWrite_WriteConflict 
    // Read Out of Bounds check
    // CHECK-NEXT: [[ENABLE:%.+]] = hw.constant true
    // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
    // CHECK-NEXT: [[MODE:%.+]] = hw.constant true
    // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
    // OOB Check
    // WRITE OOB
    // CHECK-NEXT: [[TMP0_W:%.+]] = hw.constant -4 : i4
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[TMP1_W:%.+]] = comb.icmp uge [[ADDR]], [[TMP0_W]] : i4
    // CHECK-NEXT: [[TMP2_W:%.+]] = comb.xor [[TMP1_W]], [[TRUE]] : i1
    // CHECK-NEXT: [[TMP3_W:%.+]] = comb.and [[TMP2_W]], [[ENABLE]] : i1
    // CHECK-NEXT: verif.assert [[TMP3_W]] label "write_enable" : i1
    // WRITE CONFLICT CHECK

    // CHECK-NEXT: [[SAME_ADDR1:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4
    // CHECK-NEXT: [[RW_ENABLED:%.+]] = comb.and [[ENABLE]], [[MODE]] : i1

    // CHECK-NEXT: [[BOTH_ENABLED1:%.+]] = comb.and [[ENABLE]], [[RW_ENABLED]] : i1
    // CHECK-NEXT: [[SAME_ADDR_ENABLED:%.+]] = comb.and [[SAME_ADDR1]], [[BOTH_ENABLED1]] : i1
    // Look for collisions
    // CHECK-NEXT: [[COLLISION1:%.+]] = comb.or [[SAME_ADDR_ENABLED]] : i1
    // CHECK-NEXT: [[RANDOM_DATA:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[RAND_DATA_MUX:%.+]] = comb.mux [[COLLISION1]], [[RANDOM_DATA]], %data : i20
    // CHECK-NEXT: seq.firmem.write_port %mem[[[ADDR]]] = [[RAND_DATA_MUX]], clock %clock enable [[ENABLE]] : <12 x 20>



    // READ WRITE CHECKS
    // Read Out of Bounds check
    // CHECK-NEXT: [[WRITEENABLED:%.+]] = comb.and [[MODE]], [[ENABLE]]
    // Get Read Enable
    // CHECK-NEXT: [[TRUE:%.+]] = hw.constant true
    // CHECK-NEXT: [[READMODE:%.+]] = comb.xor [[MODE]], [[TRUE]]
    // CHECK-NEXT: [[READ_ENABLED:%.+]] = comb.and [[READMODE]], [[ENABLE]]
    // WRITE OOB
    // CHECK-NEXT: [[DEPTH:%.+]] = hw.constant -4 : i4
    // CHECK-NEXT: [[OOB:%.+]] = comb.icmp uge [[ADDR]], [[DEPTH]] : i4
    // CHECK-NEXT: [[NOTOOB:%.+]] = comb.xor [[OOB]], [[TRUE]]
    // CHECK-NEXT: [[TMP4_W:%.+]] = comb.and [[NOTOOB]], [[WRITEENABLED]] : i1
    // CHECK-NEXT: verif.assert [[TMP4_W]] label "write_enable" : i1

    // WRITE CONFLICT CHECK
    // CHECK-NEXT: [[SAME_ADDR_RW1:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4
    // CHECK-NEXT: [[BOTH_ENABLED_RW1:%.+]] = comb.and [[WRITEENABLED]], [[ENABLE]] : i1
    // CHECK-NEXT: [[SAME_ADDR_ENABLED_RW1:%.+]] = comb.and [[SAME_ADDR_RW1]], [[BOTH_ENABLED_RW1]] : i1
    // Look for collisions
    // CHECK-NEXT: [[COLLISION_RW1:%.+]] = comb.or [[SAME_ADDR_ENABLED_RW1]] : i1

    // CHECK-NEXT: [[RANDOM_DATA_RW1:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[RAND_DATA_MUX_RW1:%.+]] = comb.mux [[COLLISION_RW1]], [[RANDOM_DATA_RW1]], %data : i20

    // READ OOB
    // CHECK-NEXT: [[READ_RW1:%.+]] = seq.firmem.read_write_port %mem[[[ADDR]]] = [[RAND_DATA_MUX_RW1]] if [[MODE]], clock %clock enable [[ENABLE]] : <12 x 20>
    // CHECK-NEXT: [[RAND:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[READOOB:%.+]] = comb.and [[OOB]], [[READ_ENABLED]] : i1
    // CHECK-NEXT: [[READ_MUX_RW:%.+]] = comb.mux [[READOOB]], [[RAND]], [[READ_RW1]]
    // CHECK-NEXT: hw.output [[READ_MUX_RW]] : i20
    // CHECK-NEXT: }
      hw.module @ReadWrite_WriteConflict(in %data: i20, in %clock: !seq.clock,  out z: i20) {
      %enable = hw.constant true // Set to constant 1
      %addr = hw.constant 6 : i4
      %mode = hw.constant true
      %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

        seq.firmem.write_port %mem[%addr] = %data, clock %clock enable %enable: <12 x 20>
        %0 = seq.firmem.read_write_port %mem[%addr] = %data if %mode, clock %clock enable %enable: <12 x 20>


      // Does a write have an output?
        hw.output %0 : i20
      }





  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



  
  
  // Read Write, Read Write Conflict
  //-----

    //CHECK-LABEL: hw.module @ReadWrite_ReadWriteConflict
    // Read Out of Bounds check
    // CHECK-NEXT: [[ENABLE:%.+]] = hw.constant true
    // CHECK-NEXT: [[ADDR:%.+]] = hw.constant 6 : i4
    // CHECK-NEXT: [[MODE:%.+]] = hw.constant true
    // CHECK-NEXT: %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>
    // OOB Check
    

     // READ WRITE CHECKS 1
    // Read Out of Bounds check
    // CHECK-NEXT: [[WRITE_ENABLED1:%.+]] = comb.and [[MODE]], [[ENABLE]]
    // Get Read Enable
    // CHECK-NEXT: [[TRUE1:%.+]] = hw.constant true
    // CHECK-NEXT: [[READMODE1:%.+]] = comb.xor [[MODE]], [[TRUE1]] : i1
    // CHECK-NEXT: [[READ_ENABLED1:%.+]] = comb.and [[READMODE1]], [[ENABLE]] : i1
   // WRITE OOB
    // CHECK-NEXT: [[DEPTH1:%.+]] = hw.constant -4 : i4
    // CHECK-NEXT: [[OOB1:%.+]] = comb.icmp uge [[ADDR]], [[DEPTH1]] : i4
    // CHECK-NEXT: [[NOTOOB1:%.+]] = comb.xor [[OOB1]], [[TRUE1]]
    // CHECK-NEXT: [[TMP4_W1:%.+]] = comb.and [[NOTOOB1]], [[WRITE_ENABLED1]] : i1
    // CHECK-NEXT: verif.assert [[TMP4_W1]] label "write_enable" : i1

    // WRITE CONFLICT CHECK
    // CHECK-NEXT: [[SAME_ADDR_RW1:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4
    // CHECK-NEXT: [[RW2_ENABLED_MODEW:%.+]] = comb.and [[ENABLE]], [[MODE]] : i1
    // CHECK-NEXT: [[BOTH_WR_EN:%.+]] = comb.and [[WRITE_ENABLED1]], [[RW2_ENABLED_MODEW]] : i1
    // CHECK-NEXT: [[WR_CONFLICT:%.+]] = comb.and [[SAME_ADDR_RW1]], [[BOTH_WR_EN]] : i1




    // Read-Write Conflict Check
    // CHECK-NEXT: [[RW_Read_and_Write:%.+]] = comb.and [[READ_ENABLED1]], [[RW2_ENABLED_MODEW]] : i1
    // CHECK-NEXT: [[RW_Conflict1:%.+]] = comb.and [[SAME_ADDR_RW1]], [[RW_Read_and_Write]] : i1

    // Look for collisions
    // Write Collision  : n/a
    // CHECK-NEXT: [[COLLISION_RW1:%.+]] = comb.or [[WR_CONFLICT]] : i1
    // CHECK-NEXT: [[RANDOM_DATA_RW1:%.+]] = verif.symbolic_value : i20
    // Here is issue. Relies on future.
    // CHECK-NEXT: [[RAND_DATA_MUX_RW1:%.+]] = comb.mux [[COLLISION_RW1]], [[RANDOM_DATA_RW1]], %data : i20


    // READ OOB
    // CHECK-NEXT: [[READ_RW1:%.+]] = seq.firmem.read_write_port %mem[[[ADDR]]] = [[RAND_DATA_MUX_RW1]] if [[MODE]], clock %clock enable [[ENABLE]] : <12 x 20>
  
    // CHECK-NEXT: [[RAND1:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[READOOB1:%.+]] = comb.and [[OOB1]], [[READ_ENABLED1]] : i1
    // CHECK-NEXT: [[READ_MUX_RW1:%.+]] = comb.mux [[READOOB1]], [[RAND1]], [[READ_RW1]] : i20

    // Read Collision
    // CHECK-NEXT: [[RW_Collision1:%.+]] = comb.or [[RW_Conflict1]] : i1
    // CHECK-NEXT: [[RANDOM_DATA_RW11:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[RW_READ_OUTPUT1:%.+]] = comb.mux [[RW_Collision1]], [[RANDOM_DATA_RW11]], [[READ_MUX_RW1]] : i20






     // READ WRITE CHECKS 2
    // Read Out of Bounds check
    // CHECK-NEXT: [[WRITE_ENABLED2:%.+]] = comb.and [[MODE]], [[ENABLE]]
    // Get Read Enable
    // CHECK-NEXT: [[TRUE2:%.+]] = hw.constant true
    // CHECK-NEXT: [[READMODE2:%.+]] = comb.xor [[MODE]], [[TRUE2]] : i1
    // CHECK-NEXT: [[READ_ENABLED2:%.+]] = comb.and [[READMODE2]], [[ENABLE]] : i1
   // WRITE OOB
    // CHECK-NEXT: [[DEPTH2:%.+]] = hw.constant -4 : i4
    // CHECK-NEXT: [[OOB2:%.+]] = comb.icmp uge [[ADDR]], [[DEPTH2]] : i4
    // CHECK-NEXT: [[NOTOOB2:%.+]] = comb.xor [[OOB2]], [[TRUE2]]
    // CHECK-NEXT: [[TMP4_W2:%.+]] = comb.and [[NOTOOB2]], [[WRITE_ENABLED2]] : i1
    // CHECK-NEXT: verif.assert [[TMP4_W2]] label "write_enable" : i1

    // WRITE CONFLICT CHECK
    // CHECK-NEXT: [[SAME_ADDR_RW2:%.+]] = comb.icmp eq [[ADDR]], [[ADDR]] : i4
    // CHECK-NEXT: [[RW2_ENABLED_MODEW:%.+]] = comb.and [[ENABLE]], [[MODE]] : i1
    // CHECK-NEXT: [[BOTH_WR_EN2:%.+]] = comb.and [[WRITE_ENABLED2]], [[RW2_ENABLED_MODEW]] : i1
    // CHECK-NEXT: [[WR_CONFLICT:%.+]] = comb.and [[SAME_ADDR_RW2]], [[BOTH_WR_EN2]] : i1




    // Read-Write Conflict Check
    // CHECK-NEXT: [[RW_Read_and_Write:%.+]] = comb.and [[READ_ENABLED2]], [[RW2_ENABLED_MODEW]] : i1
    // CHECK-NEXT: [[RW_Conflict2:%.+]] = comb.and [[SAME_ADDR_RW2]], [[RW_Read_and_Write]] : i1





    // Look for collisions
    // Write Collision  : n/a
    // CHECK-NEXT: [[COLLISION_RW2:%.+]] = comb.or [[WR_CONFLICT]] : i1
    // CHECK-NEXT: [[RANDOM_DATA_RW2:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[RAND_DATA_MUX_RW2:%.+]] = comb.mux [[COLLISION_RW2]], [[RANDOM_DATA_RW2]], %data : i20


    // READ OOB
    // CHECK-NEXT: [[READ_RW2:%.+]] = seq.firmem.read_write_port %mem[[[ADDR]]] = [[RAND_DATA_MUX_RW2]] if [[MODE]], clock %clock enable [[ENABLE]] : <12 x 20>
  
    // CHECK-NEXT: [[RAND2:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[READOOB2:%.+]] = comb.and [[OOB2]], [[READ_ENABLED2]] : i1
    // CHECK-NEXT: [[READ_MUX_RW2:%.+]] = comb.mux [[READOOB2]], [[RAND2]], [[READ_RW2]] : i20

    // Read Collision
    // CHECK-NEXT: [[RW_Collision2:%.+]] = comb.or [[RW_Conflict2]] : i1
    // CHECK-NEXT: [[RANDOM_DATA_RW22:%.+]] = verif.symbolic_value : i20
    // CHECK-NEXT: [[RW_READ_OUTPUT2:%.+]] = comb.mux [[RW_Collision2]], [[RANDOM_DATA_RW22]], [[READ_MUX_RW2]] : i20



    // CHECK-NEXT: hw.output [[RW_READ_OUTPUT1]] : i20
    // CHECK-NEXT: }
      hw.module @ReadWrite_ReadWriteConflict(in %data: i20, in %clock: !seq.clock,  out z: i20) {
      %enable = hw.constant true // Set to constant 1
      %addr = hw.constant 6 : i4
      %mode = hw.constant true
      %mem = seq.firmem 0, 1, undefined, undefined : <12 x 20>

        %0 = seq.firmem.read_write_port %mem[%addr] = %data if %mode, clock %clock enable %enable: <12 x 20>
        %1 = seq.firmem.read_write_port %mem[%addr] = %data if %mode, clock %clock enable %enable: <12 x 20>

      // Does a write have an output?
        hw.output %0 : i20
      }


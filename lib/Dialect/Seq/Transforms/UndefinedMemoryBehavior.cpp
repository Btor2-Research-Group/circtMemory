//===- UndefinedMemoryBehavior.cpp - .Handle Undefined Behavior -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_UNDEFINEDMEMORYBEHAVIOR
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;

namespace {

/// Lowers seq.compreg.ce to a seq.compreg with the clock enable signal
/// built into the next logic, i.e. `next := mux(clock_enable, next, current)`
struct UndefinedMemoryBehavior
: seq::impl::UndefinedMemoryBehaviorBase<UndefinedMemoryBehavior> {
public:
  void runOnOperation() override;
  
};

} // namespace

void UndefinedMemoryBehavior::runOnOperation() {
  // The two for loop place to code
}

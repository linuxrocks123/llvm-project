# MachineLaneSSAUpdater Test Framework

This directory contains MIR test scenarios for the `MachineLaneSSAUpdater` utility, a lane-aware SSA reconstruction tool for Machine IR.

## Overview

`MachineLaneSSAUpdater` is a utility class (not a pass) that performs SSA repair after:
1. **New Definition Insertion**: When a new virtual register definition is added to existing SSA form
2. **Spill/Reload**: When a register is spilled and later reloaded, requiring LiveInterval reconstruction

## Test Infrastructure

Since `MachineLaneSSAUpdater` is a **target-independent utility** (not a pass), testing is done through:

### Primary: Unit Tests

Located in `llvm/unittests/CodeGen/MachineLaneSSAUpdaterTest.cpp` (target-independent location), these tests:

- Parse MIR strings or files programmatically
- Set up required analyses (SlotIndexes, LiveIntervals, MachineDominatorTree)
- Directly call MachineLaneSSAUpdater APIs
- Use GoogleTest assertions to verify correctness

**Usage:**
```bash
ninja CodeGenTests
./unittests/CodeGen/CodeGenTests --gtest_filter=MachineLaneSSAUpdaterTest.*
```

### MIR Test Files (in this directory)

These `.mir` files serve as test scenarios that can be:
1. Parsed by unit tests to create test cases
2. Used as documentation of expected behavior
3. Potentially run with lit if a test harness is added

**Note**: These are primarily reference MIR for unit tests, not standalone lit tests.

### Test Scenarios

#### 1. Simple New Definition (`simple_new_def.mir`)

**Scenario:**
- Original vreg `%0` defined in `bb.0`
- New definition inserted in `bb.1`
- Uses in `bb.2` should be rewritten

**What's Tested:**
- `MachineLaneSSAUpdater::addDefAndRepairNewDef()`
- Use rewriting in dominated blocks
- LiveInterval correctness

**Expected Behavior:**
```
bb.0:
  %0 = IMPLICIT_DEF
  use %0              // Still uses %0

bb.1:
  %1 = V_MOV_B32 42   // New definition inserted by test pass

bb.2:
  use %1              // Rewritten to use %1
```

#### 2. Spill/Reload (`spill_reload.mir`)

**Scenario:**
- High register pressure triggers spill
- `SpillCutCollector` captures cut information
- Reload inserted later
- SSA reconstructed with proper LiveInterval extension

**What's Tested:**
- `SpillCutCollector::cut()`
- `MachineLaneSSAUpdater::addDefAndRepairAfterSpill()`
- LiveInterval cutting and extension
- Captured endpoint usage

**Expected Behavior:**
```
bb.0:
  %0 = IMPLICIT_DEF
  use %0              // Use before spill

bb.1:
  [High RP]
  SPILL %0            // Cut LiveInterval here
  
bb.2:
  %1 = RELOAD         // New vreg from reload
  
bb.3:
  use %1              // Uses rewritten to reload vreg
```

#### 3. Partial Lanes (`partial_lanes.mir`)

**Scenario:**
- Wide register (e.g., `vreg_128`) with multiple subregisters
- New definition only affects specific lanes (e.g., `sub0`)
- Uses of affected lanes should be rewritten
- Uses of other lanes should remain unchanged

**What's Tested:**
- Lane-aware SSA reconstruction
- Subregister use rewriting
- REG_SEQUENCE insertion for mixed uses
- Correct lane mask handling

**Expected Behavior:**
```
bb.0:
  %0:vreg_128 = IMPLICIT_DEF
  use %0.sub0         // sub0 before new def
  use %0.sub1         // sub1 before new def

bb.1:
  %1:vgpr_32 = V_MOV_B32 100  // New def for sub0 lanes only

bb.2:
  use %1              // sub0 use rewritten
  use %0.sub1         // sub1 still uses %0
  %2 = REG_SEQUENCE %1, sub0, %0.sub1, sub1  // Mixed use
  use %2.sub0_sub1
```

#### 4. PHI Insertion (`phi_insertion.mir`)

**Scenario:**
- Diamond control flow: `bb.0 -> {bb.1, bb.2} -> bb.3`
- New definition in one path only
- PHI must be inserted at join point

**What's Tested:**
- Automatic PHI insertion
- Per-edge lane analysis
- Dominance frontier computation
- PHI operand correctness

**Expected Behavior:**
```
bb.0:
  %0 = IMPLICIT_DEF
  branch to bb.1 or bb.2

bb.1:
  %1 = V_MOV_B32 42   // New def in this path
  branch bb.3

bb.2:
  use %0              // Original continues here
  branch bb.3

bb.3:
  %2 = PHI %1, bb.1, %0, bb.2  // PHI inserted by updater
  use %2              // Uses reference PHI result
```

## Dependencies

The utility requires these analyses to be computed:
- `SlotIndexes`: For instruction indexing
- `LiveIntervals`: For LiveInterval computation and updates
- `MachineDominatorTree`: For dominance analysis and PHI placement

## Running Tests

### Unit Tests (Primary):
```bash
# Build unit tests
ninja CodeGenTests

# Run all MachineLaneSSAUpdater tests
./unittests/CodeGen/CodeGenTests --gtest_filter=MachineLaneSSAUpdaterTest.*

# Run specific test
./unittests/CodeGen/CodeGenTests --gtest_filter=MachineLaneSSAUpdaterTest.SimpleNewDefReconstruction
```

### MIR Files:
The `.mir` files in this directory are used as input to the unit tests.
They can be parsed programmatically to create test scenarios.

## Current Test Status

**Current Implementation**: Basic smoke tests in `llvm/unittests/CodeGen/MachineLaneSSAUpdaterTest.cpp`

The current unit tests verify:
- API compilation and linking
- Basic LaneBitmask operations
- Helper function signatures

**Why Limited Tests?**

Setting up full integration tests requires substantial infrastructure:
- MIRParser for loading test MIR
- Pass Manager for running analyses
- Proper SlotIndexes, LiveIntervals, MachineDominatorTree setup
- Target-specific initialization

This is similar complexity to the [NextUseAnalysisTest](https://github.com/llvm/llvm-project/pull/156079) which has ~1000 lines of test infrastructure.

**Future Enhancement**:

Full integration tests would follow this pattern:
```cpp
TEST_F(MachineLaneSSAUpdaterTest, SimpleNewDefReconstruction) {
  // 1. Parse MIR file
  MachineFunction *MF = parseMIRFile("simple_new_def.mir");
  
  // 2. Setup analyses via pass manager
  setupAnalyses(*MF);
  
  // 3. Get analyses
  SlotIndexes &SI = ...;
  LiveIntervals &LIS = ...;
  MachineDominatorTree &MDT = ...;
  
  // 4. Exercise utility
  MachineLaneSSAUpdater Updater(*MF, LIS, SI, MDT);
  Register Result = Updater.addDefAndRepairNewDef(...);
  
  // 5. Verify
  EXPECT_TRUE(Result.isValid());
}
```

Contributions welcome for full integration testing!

## Adding New Tests

1. Create a new `.mir` file in this directory
2. Follow the naming convention to trigger appropriate test mode
3. Add CHECK directives to verify expected transformations
4. Document the test scenario in comments

Example:
```mir
# RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 \
#      -run-pass=test-machine-lane-ssa-updater %s -o - | FileCheck %s

--- |
  define amdgpu_kernel void @my_test() {
    ret void
  }
...
---
name: my_test
body: |
  bb.0:
    ; Test scenario here
...
```

## Debugging

Run unit tests with verbose output:
```bash
./unittests/CodeGen/CodeGenTests --gtest_filter=MachineLaneSSAUpdaterTest.* --gtest_verbose

# With LLVM debug output (requires debug build)
./unittests/CodeGen/CodeGenTests --gtest_filter=MachineLaneSSAUpdaterTest.* \
    --debug-only=machine-lane-ssa-updater
```

## Reference

See the [NextUseAnalysisTest](https://github.com/llvm/llvm-project/pull/156079) for a similar unit testing pattern.

## Implementation Notes

- **Target Independence**: Tests are in `llvm/unittests/CodeGen/` (NOT target-specific)
- Uses AMDGPU target for testing due to rich subregister structure, but utility is target-independent
- Unit tests use MIRParser to load test scenarios from `.mir` files
- Can be extended for other targets (X86, ARM, etc.) with similar subregister needs


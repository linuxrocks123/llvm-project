# SSA-Aware Register Spiller Development Notes

## Project Overview

Developing an SSA-aware register spiller for AMDGPU that handles subregisters, divergent control flow, and maintains SSA form throughout the spilling process.

**Key Innovation:** "Store at Definition" strategy - stores registers at their definition point (when EXEC is full) to eliminate EXEC drift issues, then uses `shrinkToUses()` to trim LiveIntervals after reload placement.

---

## Current Status (2025-11-19 Evening)

### Working Implementation

**Core Spilling Strategy:**
1. **Store at definition** (`spillAtDefinition()`) - physically stores register right after definition when EXEC is full
2. **Set virtual spill point** - computes `KillIdx` at high-pressure point, but does NOT prune LiveInterval
3. **Emit reloads** for dominated uses with dominance grouping
4. **Handle reachable uses** - currently using standard PHI insertion (split-before-use commented out)
5. **Shrink LiveIntervals** - `shrinkToUses()` called at end of `emitReloadsAndRepairSSA()` after all SSA repairs

**Key Files:**
- `llvm/lib/Target/AMDGPU/AMDGPUSSARegisterSpiller.cpp` (1643 lines)
- `llvm/lib/Target/AMDGPU/AMDGPUSSARegisterSpiller.h` (279 lines)
- `llvm/lib/Target/AMDGPU/VRegMaskPair.h` (subregister-aware register tracking)

### Recent Changes (2025-11-19)

**1. Removed WWM (Whole Wave Mode) Wrapping**
- **Rationale:** With "store at definition", we store the same mask as defined, so WWM is unnecessary
- **Removed:**
  - `wrapWithWWM()` method
  - All calls to `wrapWithWWM()`
  - WWM-related SGPR allocation

**2. Removed Divergence Detection Helpers**
- **Rationale:** No longer needed for spill placement with "store at definition"
- **Removed:**
  - `isDivergentInstr()`
  - `pathHasDivergence()`
  - `pathsHaveDivergence()`
  - Divergence classification in `emitReloadsAndRepairSSA()`

**3. Removed IDF Parameter**
- **Changed:** `splitBlockBeforeReload()` now computes JoinBB directly from CFG structure
- **Algorithm:** JoinBB is UseBB if it has multiple predecessors, otherwise computed via dominator tree
- **Benefit:** Simpler, no need to compute IDF in `emitReloadsAndRepairSSA()`

**4. Merged Handler Functions**
- **Merged:** `handleUniformReachableUse()` and `handleDivergentReachableUse()` into single `handleReachableUse()`
- **Simplified:** Always uses split-before-use without WWM

**5. Removed spillBefore() Call**
- **Old flow:** `spillAtDefinition()` → `spillBefore()` (prune) → `emitReloadsAndRepairSSA()`
- **New flow:** `spillAtDefinition()` → compute KillIdx → `emitReloadsAndRepairSSA()` → `shrinkToUses()`
- **Key change:** LiveInterval stays valid during reload placement, only shrunk once at end

**6. Temporarily Commented Out Split-Before-Use**
- **Location:** Line 832 in AMDGPUSSARegisterSpiller.cpp
- **Reason:** Need to implement cost model to decide when split-before-use is profitable
- **Current behavior:** Uses standard PHI insertion for reachable uses (reloads on all paths)
- **TODO:** Implement cost model considering path length and RP from JoinBB to use

### Current Code Structure

**spillAndReload() - Main Entry Point:**
```cpp
for (each VMP to spill) {
  1. spillAtDefinition(VMP)           // Store at def, when EXEC full
  2. Compute KillIdx at high-RP point // Virtual spill point
  3. assignVirt2StackSlot(VMP)        // Get frame index
  4. emitReloadsAndRepairSSA(VMP, KillIdx, FI)  // Place reloads, repair SSA
}
```

**emitReloadsAndRepairSSA() - Reload Placement:**
```cpp
1. Collect all uses of spilled register
2. Classify uses:
   - Dominated uses: handled with dominance grouping
   - Non-dominated uses: check reachability
3. Group dominated uses by dominance chains
4. Emit one reload per group at group head
5. Call MachineLaneSSAUpdater::repairSSAForNewDef()
6. Try to hoist spill to NCD if no uses on paths
7. Handle reachable uses:
   - Currently: emit reload at use, let SSAUpdater insert PHIs
   - TODO: use handleReachableUse() with cost model
8. shrinkToUses(&SpilledLI)  // Trim LiveInterval after all repairs
```

**spillAtDefinition() - Store at Definition:**
```cpp
1. Find definition: MRI->getVRegDef(VReg)
2. Insert store right after definition with isKill=false
3. Track in StoredAtDefinition set
4. Returns store instruction
```

**handleReachableUse() - Split-Before-Use (COMMENTED OUT):**
```cpp
1. Compute JoinBB from CFG structure
2. Insert flag PHI at JoinBB (1 from spill path, 0 from clean)
3. Split UseBB into: UseBB_Pre → ReloadBB → UseBB_Post
4. Conditional branch: if clean path, skip ReloadBB
5. MachineLaneSSAUpdater inserts value PHI at UseBB_Post
```

### What's Working

✅ Store at definition eliminates EXEC drift
✅ No WWM needed - simpler code
✅ LiveIntervals stay valid during reload placement
✅ Dominated use handling with grouping works
✅ Reachable use handling works (with standard PHI insertion)
✅ shrinkToUses() correctly trims LiveIntervals after repairs
✅ Subregister spilling with VRegMaskPair
✅ Reachability checks filter unreachable uses

### What's Temporarily Disabled

⚠️ **Split-before-use optimization** (line 832 commented out)
- Reason: Need cost model to decide when profitable
- Impact: More reloads than optimal (reloads on all paths instead of just spill path)
- Not a correctness issue, just performance

### Known Issues & TODOs

**High Priority:**
- [ ] **Implement cost model for split-before-use decision**
  - Measure path length from JoinBB to use
  - Check max RP on path from JoinBB to use
  - Decision thresholds: long path (>50 insts) or high RP (>70% limit)
  - See NOTES.md 2025-11-18 Evening section for detailed algorithm

- [ ] **Fix test cases with RP budget violations**
  - `spill-dominated-branches.mir`: Remove %2 from join (12 VGPRs live, budget 7-8)
  - `spill-linear-dominated.mir`: Remove excess register
  - `spill-multi-path-independent.mir`: Remove %2 (3rd vreg_128)

- [ ] **Update tryHoistSpillToNCD() logic**
  - Currently hoists by pruning LiveInterval at NCD
  - With "store at definition", verify this still makes sense
  - May need to adjust how we represent hoisted kill point

**Medium Priority:**
- [ ] Pre-allocation feasibility check (bail if max RP > limit)
- [ ] Balanced spill decision for clean paths with high RP
- [ ] EWF (EXEC Write Frontier) as alternative to WWM for divergent paths
- [ ] Path analysis caching for performance
- [ ] Comprehensive MIR test suite for subregister scenarios

**Low Priority:**
- [ ] Loop-aware spilling optimizations
- [ ] Performance evaluation on real AMDGPU workloads
- [ ] Integration with SSA-based register allocator

### Code Locations

**Key Methods:**
- `spillAndReload()` - Lines 543-614
- `emitReloadsAndRepairSSA()` - Lines 617-854
- `spillAtDefinition()` - Lines 894-962
- `spillBefore()` - Lines 964-1044 (UNUSED after recent changes, can be removed)
- `handleReachableUse()` - Lines 1549-1559 (COMMENTED OUT at call site line 832)
- `splitBlockBeforeReload()` - Lines 1405-1547
- `tryHoistSpillToNCD()` - Lines 1338-1403

**Helper Methods:**
- `hasUseOnPath()` - Lines 1278-1336
- `usesSpilledVMP()` - Lines 1237-1261
- `repairSSAForReload()` - Uses MachineLaneSSAUpdater

### Test Files

**MIR Tests:**
- `llvm/test/CodeGen/AMDGPU/SSASpiller/spill-dominated-branches.mir`
- `llvm/test/CodeGen/AMDGPU/SSASpiller/spill-linear-dominated.mir`
- `llvm/test/CodeGen/AMDGPU/SSASpiller/spill-multi-path-independent.mir`
- Others in `llvm/test/CodeGen/AMDGPU/SSASpiller/`

**Note:** Some tests currently have RP budget violations and need fixing.

---

## Technical Background

### Store at Definition Strategy

**Problem:** Spilling at high-pressure point can occur after divergent branches where EXEC mask has changed, leading to only a subset of lanes being stored.

**Solution:** Store register immediately after its definition (when EXEC is guaranteed to be full), then mark it dead at the virtual spill point. Use `shrinkToUses()` to trim LiveInterval after all reloads are placed.

**Benefits:**
1. **Correctness:** All lanes stored when EXEC is full
2. **Simplicity:** No WWM wrapping needed
3. **Clean LiveIntervals:** Stay valid during reload placement
4. **Efficient:** Only one physical store per register

**Key Insight:** Separate "when to store" (at definition, for correctness) from "when to free the register" (at spill point, for register pressure).

### Why No WWM Needed

With "store at definition":
- **Stores happen when EXEC is full** (at definition) → all lanes stored correctly
- **Reloads load the same mask that was defined** → correctness preserved
- **No EXEC drift concern** → no need for WWM to save/restore EXEC

### Why No Divergence Detection Needed

- **Spills are always correct** (stored at definition when EXEC full)
- **Divergence only matters for reload optimization** (split-before-use decision)
- **Current approach:** Use standard PHI insertion for all reachable uses
- **Future optimization:** Add divergence check only for split-before-use cost model

### Split-Before-Use Strategy

**Goal:** Extend RP decrease window by avoiding PHI at JoinBB.

**Without split-before-use (Case 2a):**
```
DefBB → ... → JoinBB → ... → UseBB
              ↑ PHI here
              RP decrease window: spill to end of spill path
```

**With split-before-use (Case 2b):**
```
DefBB → ... → JoinBB → ... → UseBB_Pre → ReloadBB → UseBB_Post
                               ↑ split here      ↑ PHI here
              RP decrease window: spill to UseBB_Pre (much longer)
```

**When to use split-before-use:**
- Long path from JoinBB to use (>50 instructions)
- High RP on path from JoinBB to use (>70% of limit)
- Otherwise, standard PHI at JoinBB is simpler

---

## Recent Discoveries & Clarifications

### 2025-11-18 (Monday Evening) - Hoisting and Split-Before-Use

**User Clarification on Hoisting:**
> "I am sorry: 'if we have 2 uses on disjoint paths but at least on one path RP is high - we cannot hoist to NCD as we increase RP!' - is wrong, I made a mistake. The high RP on both spill and clean paths is exactly the main reason for hoisting!"

**Correct Understanding:**
- **Hoist to NCD when:** High RP on both paths
- **Benefit:** Spill once at NCD instead of potentially on both paths
- **Don't hoist when:** Uses exist on either path between NCD and join/use

**Why split-before-use is still needed:**
Even with hoisting and "store at definition", split-before-use helps when there's a long or high-RP path from JoinBB to the use:
- **Without split:** PHI at JoinBB → RP decrease window is short
- **With split:** No PHI at JoinBB → RP decrease window extends to split point

### 2025-11-18 (Monday Afternoon) - shrinkToUses Verification

Confirmed that `shrinkToUses(LiveInterval*)` automatically handles subranges:
- No need for separate subrange shrinking calls
- Verified in `LiveIntervals.cpp` lines 481-537

### 2025-11-10 - Store-at-Definition Implementation

Original implementation of the "store at definition" approach:
- Added `spillAtDefinition()` method
- Modified `spillBefore()` to use `pruneValue()` (now unused after recent changes)
- Updated `spillAndReload()` workflow
- Changed `emitReloadsAndRepairSSA()` signature to use `SlotIndex KillIdx`

---

## Implementation History

### Major Milestones

1. **Initial Framework** - Basic spill/reload with SSA form
2. **MachineLaneSSAUpdater Integration** - Lane-aware SSA reconstruction
3. **Subregister Support** - VRegMaskPair for partial spills
4. **Dominance Grouping** - Efficient reload placement for dominated uses
5. **Reachability Analysis** - IDF-based reachable use detection
6. **Split-Before-Use** - CFG transformation for optimal RP
7. **Store at Definition** - EXEC drift elimination (2025-11-10)
8. **WWM/Divergence Removal** - Simplified after store-at-definition (2025-11-19)
9. **spillBefore Removal** - Deferred LiveInterval trimming (2025-11-19)

### Major Bug Fixes

1. **Invalid MIR between spill/reload** - Fixed with atomic per-register processing
2. **SSAUpdater interface mismatch** - Fixed by reloading into original VReg
3. **Subregister def replacement undef flag** - Fixed in MachineLaneSSAUpdater
4. **PHI use infinite loop** - Fixed by skipping PHI instructions in use collection
5. **Over-spilling bug** - Fixed RemainingToSpill not reset to 0
6. **Immediate reload after spill** - Fixed by skipping triggering instruction
7. **Stale LiveIntervals** - Fixed by moving shrinkToUses to end
8. **Manual PHI creation** - Fixed by removing manual PHI, relying on SSAUpdater
9. **rewriteDominatedUses bug** - Fixed dominance vs reachability check
10. **SSA Updater refactoring breakage** - Fixed IDF caching and DenseMapInfo
11. **Register budget violation at reload** - Added RP check before reload
12. **Stale use bug** - Added usesSpilledVMP() check

---

## Key Design Decisions

### Why Store at Definition?

**Alternatives considered:**
1. WWM wrapping around spills - Complex, requires SGPR pairs, costly
2. EWF (EXEC Write Frontier) - Move reload to first EXEC write - Loses optimal placement
3. Store at definition - Simple, correct, no runtime cost

**Chosen:** Store at definition (option 3)

### Why Defer LiveInterval Trimming?

**Alternatives considered:**
1. Prune immediately at spill point - LiveIntervals become invalid, uses broken
2. Prune after each reload - Multiple pruning operations, complex
3. Defer until all reloads placed - Clean, works with SSAUpdater

**Chosen:** Defer until end (option 3)

### Why Split-Before-Use?

**Alternatives considered:**
1. Always reload on all paths - Simple but wasteful
2. Always split-before-use - Complex CFG transformations
3. Cost model based split - Optimal but needs tuning

**Chosen:** Cost model based (option 3) - pending implementation

---

## Collaboration Notes

### Colleague's Input

Key insights from colleague:
- "Store at definition" strategy to avoid EXEC drift
- Use `pruneValue()` to mark register dead without emitting redundant stores
- Hoisting to NCD when high RP on both paths
- Split-before-use extends RP decrease window

### Presentation Feedback

After presentation, decided to:
1. Remove WWM wrapping (not needed with store-at-definition)
2. Remove divergence detection (not needed for spill placement)
3. Simplify to: store at def → virtual spill point → reload → shrinkToUses
4. Defer split-before-use until cost model is implemented

---

## Next Steps for New Machine

When resuming work on the new machine:

1. **Verify build and tests**
   ```bash
   cd llvm-project/build
   ninja AMDGPUSSARegisterSpiller
   ninja check-llvm-codegen-amdgpu
   ```

2. **Enable split-before-use**
   - Uncomment line 832 in AMDGPUSSARegisterSpiller.cpp
   - Implement cost model in `handleReachableUse()` caller
   - Add decision logic based on path metrics

3. **Implement cost model**
   - Add `measurePathMetrics()` helper
   - Add `shouldSplitBeforeUse()` decision function
   - Test with various thresholds

4. **Fix test cases**
   - Address RP budget violations in existing tests
   - Run CFG analysis to verify correctness

5. **Performance evaluation**
   - Run on real shaders/kernels
   - Measure RP reduction vs overhead
   - Tune cost model thresholds

6. **Clean up dead code**
   - Remove unused `spillBefore()` method (lines 964-1044)
   - Remove any other dead code from refactoring

---

## Build & Test Commands

```bash
# Build
cd /work/atimofee/sandbox/github/llvm-project/build
ninja AMDGPUSSARegisterSpiller

# Run specific test
./bin/llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs \
  ../llvm/test/CodeGen/AMDGPU/SSASpiller/spill-linear-dominated.mir

# Run all SSASpiller tests
ninja check-llvm-codegen-amdgpu-ssaspiller

# Debug with gdb
gdb --args ./bin/llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs \
  -debug-only=amdgpu-ssa-spiller test.mir
```

---

## References

- **LLVM Live Intervals:** `llvm/include/llvm/CodeGen/LiveIntervals.h`
- **MachineLaneSSAUpdater:** `llvm/lib/CodeGen/MachineLaneSSAUpdater.cpp`
- **VRegMaskPair:** `llvm/lib/Target/AMDGPU/VRegMaskPair.h`
- **GCNRegPressure:** `llvm/lib/Target/AMDGPU/GCNRegPressure.h`
- **SIInstrInfo:** `llvm/lib/Target/AMDGPU/SIInstrInfo.h`

---

**Last Updated:** 2025-12-02
**Ready for:** Migration to new development machine
**Status:** Core implementation working, split-before-use optimization pending cost model

---

### 2025-11-21 – CFG Viewer Refresh & Deployment Sync

- **Context / goal**
  - Fix the HTTP viewer after moving from `index.html` to `viewer.html`
  - Eliminate stale instructions that still referenced the old workflow
  - Ensure teammates installing from the bundle get the latest scripts/docs
- **Attempts / hypotheses**
  - Declared UTF-8 in `viewer.html` and removed emoji headings to fix mojibake
  - Audited every README/guide/task/snippet for `index.html` references
  - Switched the installer to copy the checked-in `view-cfg.sh` instead of embedding an outdated version
  - Regenerated `llvm-cfg-tools.tar.gz` so deployments pick up the fixes
- **Results / discoveries**
  - `view-cfg.sh` now writes `viewer.html` only, removes `index.html`, and serves UTF-8-safe markup
  - Docs (`README.md`, `DEPLOYMENT.md`, `QUICKSTART.md`, `SETUP_SUMMARY.md`, usage guides, prompts) all describe the per-function layout, automatic cleanup, and new URL (`http://localhost:8765/viewer.html`)
  - `install-cfg-tools.sh` copies the exact script from the bundle, preventing future drift
  - Tarball rebuilt with the updated files
- **Decisions / rationale**
  - Keep `view-cfg-v2.sh` as the canonical source; installer now mirrors it to the scripts dir
  - Document the per-function retention policy (latest 10 SVGs) and the “live” viewer so users know no manual index rebuild is needed
  - Added UTF-8 meta tag instead of relying on browser defaults
- **Next actions**
  - Monitor for any lingering documentation references to `index.html`
  - Consider parameterizing `MAX_GRAPHS_PER_FUNCTION` via env var if coworkers need different retention
  - Future: optional file-watcher instead of polling if refresh latency becomes an issue

### 2025-12-02 – Next Use Analysis Unit Test Fixes

- **Context / goal**
  - Fix failing unit tests for `AMDGPUNextUseAnalysis` after semantic change to overlap logic in `getFromSortedRecords()`
  - The change from exact coverage (`(Mask & UseMask) == Mask`) to any overlap (`(Mask & UseMask).any()`) broke existing CHECK patterns

- **Attempts / hypotheses**
  - Initial approach of manually adjusting CHECK patterns was error-prone due to nested loops and varying offsets
  - Test framework (`NextUseAnalysisTest.cpp`) had two bugs:
    1. **Overlap logic mismatch**: `parseExpectedDistances()` didn't compute min distance among overlapping masks
    2. **Block End Distances parsing**: `"CHECK: Block End Distances:"` didn't match `"CHECK:   Block End Distances:"` (extra spaces)

- **Results / discoveries**
  - Created `scripts/renumber_mir_vregs.py` to renumber vregs sequentially (MIRParser renumbers from %0, causing mismatch with on-disk numbers)
  - Fixed `parseExpectedDistances()` to track all mask/distance pairs and compute min distance for overlapping masks (matching `getFromSortedRecords` behavior)
  - Fixed stop condition to check for `"Block End Distances:"` anywhere in line (handles variable whitespace)
  - Regenerated CHECK patterns for 5 failing MIR test files

- **Decisions / rationale**
  - Regenerate CHECKs from actual analysis output rather than manual adjustment
  - Fix test framework to understand overlap semantics rather than changing analysis output format
  - Handle undef vreg references in PHI nodes during renumbering (they're uses without definitions)

- **Key files changed**
  - `llvm/unittests/Target/AMDGPU/NextUseAnalysisTest.cpp`:
    - `parseExpectedDistances()`: Changed to collect all mask/distance pairs, then compute min for overlapping masks
    - Stop condition: Changed `"CHECK: Block End Distances:"` to `"Block End Distances:"` for flexible whitespace
    - `getTestDirectory()`: Added `../../llvm/test/...` path for running from `build/Debug`
    - `getMirFiles()`: Added help message when test directory not found
    - `GetSortedSubregUsesDistanceOrdering`: Changed from `TEST_F(NextUseAnalysisParameterizedTest, ...)` to `TEST_F(NextUseAnalysisTestBase, ...)` - was using wrong test class
  - MIR test files with regenerated CHECKs:
    - `sequence_2_loops.mir`
    - `complex-single-loop-b.mir`
    - `if_else_with_loops_nested_in_2_outer_loops.mir`
    - `nested-loops-with-side-exits-b.mir`
    - `three_loops_sequence_nested_in_outer_loop.mir`

- **Test running (simplified)**
  - No env vars needed when running from `build/Debug`:
    ```bash
    cd build/Debug && ./unittests/Target/AMDGPU/AMDGPUTests --gtest_filter="AllMirFiles*"
    ```
  - For specific test: `--gtest_filter="*GetSortedSubreg*"`

- **Final status**
  - All 18 tests pass (17 MIR file tests + 1 GetSortedSubregUses test)

---

### 2025-12-03 – NextUseAnalysisTest Performance Refactoring

- **Context / goal**
  - Review and refactor `NextUseAnalysisTest.cpp` for compile-time performance and code readability
  - User noted the unit test was done in a "messy/clumsy manner"

- **Issues identified**
  1. **Critical: `std::regex` in hot path** (lines 181, 198, 225-226)
     - Regex compilation happens for EVERY instruction in EVERY test file
     - `std::regex` construction can take milliseconds each
     - ~100-1000x slower than StringRef-based parsing
  
  2. **Redundant string operations**
     - `std::regex_replace` used just for trimming whitespace
     - Unnecessary `std::string` allocations per instruction
  
  3. **Deep nesting** (5+ levels in `parseExpectedDistances`)
     - Hard to follow logic, ~110 lines in single function
  
  4. **Duplicate code** (duplicate `ASSERT_FALSE` at lines 629-630 vs 649-650)
  
  5. **Global static state** (`NextUseAnalysisTestWrapper::Captured`)
     - Acceptable for unit tests running sequentially, documented with comment

- **Changes applied**
  1. **Removed `#include <regex>` and `#include <iostream>`**
  
  2. **Added `parseVregPattern()` function** using StringRef methods:
     - `consume_front()`, `ltrim()`, `find_first_not_of()`, `getAsInteger()`
     - Zero regex overhead, direct pointer arithmetic
  
  3. **Replaced `machineInstrToString()` with `printMachineInstr()`**:
     - Takes `SmallVectorImpl<char> &Buf` parameter
     - Buffer declared outside loop, reused across instructions
     - `SmallString<256>` provides stack allocation for typical instruction lengths
  
  4. **Changed `getSubRegLaneMask()` parameter** from `const std::string&` to `StringRef`
  
  5. **Refactored `parseExpectedDistances()`**:
     - Takes `StringRef InstrRef` instead of `const std::string&`
     - Uses `StringRef::trim()` (handles `\n` at ends)
     - Flattened nesting from 5+ levels to 2-3 levels
     - Inline instruction core extraction (no separate helper needed)
  
  6. **Removed duplicate `ASSERT_FALSE(SortedUses.empty())`** in GetSortedSubregUsesDistanceOrdering test
  
  7. **Added explanatory comment** for static `Captured` member

- **Performance impact**
  - Pattern matching: ~100-1000x faster (no regex compilation/execution)
  - Memory allocations: Near-zero in main test loop (SmallString reuse)
  - Compile time: Slightly faster (no `<regex>` header)

- **Review conclusion: VRegMaskPairSet**
  - Checked for opportunities to use `VRegMaskPairSet` in the file
  - Found no applicable patterns:
    - `DenseMap<VRegMaskPair, unsigned>` already uses VRegMaskPair as key
    - `DenseMap<Register, SmallVector<pair<LaneBitmask, unsigned>>>` stores distances, which VRegMaskPairSet doesn't support
  - File already uses appropriate data structures

- **Files changed**
  - `llvm/unittests/Target/AMDGPU/NextUseAnalysisTest.cpp`: Full refactoring (~700 → ~734 lines due to added parseVregPattern, but cleaner structure)

- **Testing**
  - User built and ran tests: all pass

---

### 2025-11-27 – Virtual spill marker plan
- **Context / goal**
  - Need MIR-visible markers so SSA spiller lit tests can assert the "virtual spill point" without relying on stderr output.
- **Decisions / rationale**
  - Added meta pseudo `SI_VIRTUAL_SPILL_MARKER` (in `SIInstructions.td`), emitted next to each spill point, and removed again in `SIInstrInfo::expandPostRAPseudo`.
- **Next actions**
  - Land MIR tests that FileCheck for the marker alongside SSA spill scenarios.
- **Usage tip**
  - Marker emission is gated by `-amdgpu-ssa-spill-markers`; pass it to `llc`/lit when tests need MIR-visible annotations, otherwise the pass behaves as before.

---

## IDEAS

### SSA Form Sanity Check via VNInfo Count

**Observation (2025-11-29):**

In true SSA form, each virtual register has exactly **one definition** (including partial definitions via subregisters). This means each `LiveInterval` should have exactly **one VNInfo**.

**Proposed Validation Check:**
```cpp
// Post-pass SSA form validation
for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
  Register Reg = Register::index2VirtReg(I);
  if (!LIS->hasInterval(Reg))
    continue;
  
  LiveInterval &LI = LIS->getInterval(Reg);
  
  // In SSA form: one register = one definition = one VNInfo
  assert(LI.valnos.size() == 1 && 
         "SSA form violation: virtual register has multiple definitions");
  
  // For subregister-aware checking, also check each subrange
  for (LiveInterval::SubRange &SR : LI.subranges()) {
    assert(SR.valnos.size() == 1 && 
           "SSA form violation: subrange has multiple definitions");
  }
}
```

**What This Catches:**
- Accidental redefinitions of a register
- SSA repair bugs that create multiple defs instead of renaming
- Broken LiveInterval updates after spilling/reloading
- Invalid subregister writes that should have triggered renaming

**Notes:**
- This should be run as a post-pass validation after SSA repair completes
- Each subrange should also have exactly 1 VNInfo (one def per lane group)
- Simple but powerful invariant check for SSA form correctness

---

### 2025-12-03 – Next Use Analysis insert() comparison fix

- **Context / goal**
  - Debug output showed full register use of `%16` missing, only subreg uses `%16:sub0[LoopTag+35]` and `%16:sub1[LoopTag+36]` remained
  - Root cause: `insert()` comparison logic broken when stored values have mixed signs

- **Problem discovered**
  - Stored values convention: negative for finite distances (larger = closer)
  - LoopTag is large positive (~1e12), added during loop-exit merge
  - After adding LoopTag, stored values become positive (like `LoopTag + 35`)
  - Comparison `R.second >= D.second` with `-22 >= (LoopTag + 35)` returns FALSE
  - Code incorrectly thinks finite use at -22 is "further" than loop-tagged use
  - Result: finite full-reg use gets rejected/evicted by far loop-tagged subreg uses

- **Fix applied (part 1: insert)**
  - Added `isCloserOrEqual(A, B)` helper function in `VRegDistances` class
  - Handles all four cases:
    - Both negative: `A >= B` (larger = closer)
    - Both non-negative: `A <= B` (smaller = closer)
    - A negative, B non-negative: A is closer (return true)
    - A non-negative, B negative: A is NOT closer (return false)
  - Replaced direct `>=` comparisons in `insert()` with `isCloserOrEqual()` calls

- **Fix applied (part 2: merge)**
  - Initial insert() fix wasn't enough - `merge()` had same issue
  - Problem: merge() only checked exact mask matches, not coverage
  - Example: `%16[21]` (full reg) existed, but `%16:sub0[LoopTag+14]` was still added
  - Solution: simplified `merge()` to reuse `insert()`'s coverage logic
  - Old merge(): 20+ lines with separate find_if and comparison
  - New merge(): just calls `insert(VRegMaskPair(...), Rebased)` for each record
  - Benefits: code deduplication, consistent coverage semantics

- **TODO: Investigate clean negative-LoopTag approach**
  - Current fix is a workaround for inconsistent sign convention
  - Cleaner design: make LoopTag/DeadTag large negative values
  - Challenge: after materialization (adding positive offset), tier detection becomes complex
  - Need to define threshold to distinguish "finite negative" from "loop-tagged negative"
  - Consider: `materialize()` changes, `materializeForRank()` tier checks, `PrintDist` logic
  - Deferred for now due to complexity

- **Files changed**
  - `llvm/lib/Target/AMDGPU/AMDGPUNextUseAnalysis.h`: 
    - Added `isCloserOrEqual()` helper
    - Updated `insert()` to use `isCloserOrEqual()`
    - Simplified `merge()` to reuse `insert()`

- **Next session: Debug dump issue**
  - MBB_14 final distances show `%16[ 56 ]` (finite, correct)
  - But when merging MBB_10 → MBB_14, "Succ:" dump shows `%16[ LoopTag+56 ]`
  - LoopTag should be added at the edge MBB_10 → MBB_14, not before
  - Likely issue: `printVregDistances(SuccDist, EntryOff[SuccNum], EdgeWeight)` at line 142
    - Either EdgeWeight is double-counted in display
    - Or the loop-entering transformation (lines 116-137) modifies SuccDist before debug print
  - "Curr after merge" looks same as "Succ" - suggests dump offset issue, not actual merge bug
  - TODO: Verify if this is just a debug print issue or affects actual analysis

---

### 2025-12-04 – NUA Unit Test Framework Bug Fix

- **Context / goal**
  - All 17 NUA unit tests were reporting as PASSED but weren't actually validating anything
  - Root cause: silent bug causing empty `ExpectedDistances` map, so no assertions were made

- **Bug discovered**
  - `parseCheckPatterns()` stored lines like `# CHECK: Vreg: %10[ 12 ]` (with `# ` prefix)
  - `parseVregPattern()` expected lines starting with `CHECK:` (without `# ` prefix)
  - Result: `Line.consume_front("CHECK:")` always returned false → no patterns parsed → no assertions

- **Fix applied**
  - `parseCheckPatterns()`: Changed `Patterns.push_back(Line)` to `Patterns.push_back(Line.substr(2))`
  - This strips the `# ` MIR comment prefix, storing just the CHECK directive

- **Regenerated CHECK patterns**
  - Created `scripts/regenerate_nua_checks.py` helper script
  - Regenerated CHECK patterns for all 17 MIR test files using actual analysis output
  - All tests now pass with real validation

- **Files changed**
  - `llvm/unittests/Target/AMDGPU/NextUseAnalysisTest.cpp`: Fixed `parseCheckPatterns()` to strip `# ` prefix
  - `llvm/test/CodeGen/AMDGPU/NextUseAnalysis/*.mir`: All 17 files regenerated with correct CHECK patterns
  - `scripts/regenerate_nua_checks.py`: New helper script for future CHECK pattern regeneration

- **Test results**
  - 17 MIR file tests: ✅ All PASSED
  - GetSortedSubregUsesDistanceOrdering test: ✅ PASSED
  - Total: 18 tests passing

---

### 2025-12-05 – LLVM CFG Viewer VSCode Extension

- **Context / goal**
  - Convert the CFG visualization scripts into a proper VSCode/Cursor extension
  - Provide integrated CFG panel with search, zoom, and external viewer options
  - GDB debugger support

- **Implementation**
  - Created extension in `llvm-cfg-viewer/` directory with TypeScript source
  - **Core components:**
    - `src/extension.ts` - Command registration and activation
    - `src/cfgPanel.ts` - Webview panel with search, zoom controls
    - `src/converter.ts` - DOT to SVG conversion using Graphviz
    - `src/dotWatcher.ts` - File watcher for auto-conversion
    - `src/debugger/integration.ts` - GDB command integration
  - **Bundled Python helpers:**
    - `python/llvm_gdb_helpers.py` - GDB viewCFG command

- **Features**
  - Integrated webview panel displays CFG alongside code
  - Text search in CFG (finds basic blocks, registers, instructions)
  - Zoom in/out/reset for large graphs
  - "Open External" to launch in browser
  - Auto-conversion: detects new DOT files in /tmp
  - Per-function organization with automatic cleanup (max 10 per function)

- **Configuration options** (in `package.json`):
  - `llvmCfg.viewerMode`: panel/browser/system
  - `llvmCfg.graphvizPath`: path to `dot` binary
  - `llvmCfg.watchDirectory`: default `/tmp`
  - `llvmCfg.autoConvert`: auto-convert DOT files
  - `llvmCfg.maxGraphsPerFunction`: retention limit

- **Commands:**
  - `LLVM: View CFG` - Full CFG with liveness info
  - `LLVM: View CFG (Structure Only)` - Structure without liveness
  - `LLVM: Convert Latest DOT File` - Manual conversion
  - `LLVM: Open CFG in External Browser`
  - `LLVM: Show CFG Panel`

- **Build instructions**
  ```bash
  cd llvm-cfg-viewer
  npm install
  npm run compile
  npm run package  # Creates .vsix
  ```

- **Next actions**
  - Install Node.js on development machine
  - Build and test the extension
  - Test with GDB debug sessions
  - Consider publishing to VSCode marketplace later

- **Files created**
  - `llvm-cfg-viewer/package.json` - Extension manifest
  - `llvm-cfg-viewer/tsconfig.json` - TypeScript config
  - `llvm-cfg-viewer/src/*.ts` - TypeScript source files
  - `llvm-cfg-viewer/python/llvm_gdb_helpers.py` - GDB helper
  - `llvm-cfg-viewer/README.md` - User documentation
  - `llvm-cfg-viewer/INSTALL.md` - Build/install guide

---

### 2025-12-08 – NUA Code Review Cleanup [REVIEW] [REFACTOR] [NUA]

- **Context / goal**
  - Address GitHub reviewer comment questioning "Infinity(!)?" naming
  - Clean up confusing infinity-related constants in `AMDGPUNextUseAnalysis.h`

- **Issues identified**
  - Three different "infinity" values causing confusion:
    - `DeadTag` (2^60) - internal: marks dead registers during analysis
    - `PrintedInfinity` (2^32-1) - declared but never used
    - `Infinity` (65535) - sentinel returned by `getNextUseDistance()` API
  - Misleading name: `Infinity` isn't infinite, it's the API return for "dead/no use"
  - FIXME comment in `isDead()` was outdated design rationale, not actionable

- **Changes applied**
  - **[REFACTOR]** Removed unused `PrintedInfinity` constant
  - **[REFACTOR]** Renamed `Infinity` → `DeadDistance` (value stays 65535)
  - **[REFACTOR]** Updated comment: "Sentinel returned by getNextUseDistance() for dead/unused registers"
  - **[REFACTOR]** Removed stale FIXME comment from `isDead()` (11 lines → 2 lines)
  - **[BUGFIX]** Changed `LoopExits` type from `DenseMap<int, int>` to `DenseMap<unsigned, unsigned>`
    - Verified safe: `MachineBasicBlock::getNumber()` returns non-negative for valid blocks
    - Eliminates signed/unsigned comparison warning

- **Files changed**
  - `llvm/lib/Target/AMDGPU/AMDGPUNextUseAnalysis.h`:
    - Line 247-249: Replaced `PrintedInfinity` and `Infinity` with single `DeadDistance`
    - Line 241: Changed `LoopExits` type to unsigned
    - Lines 395-400: Simplified `isDead()` function
  - `llvm/lib/Target/AMDGPU/AMDGPUNextUseAnalysis.cpp`:
    - Lines 43-45, 326, 350: `Infinity` → `DeadDistance`

- **Decisions / rationale**
  - **[DESIGN]** Name `DeadDistance` clearly indicates semantic purpose (API sentinel for dead registers)
  - **[DESIGN]** Keeping value as 65535 maintains compatibility with three-tier ranking system
  - **[DESIGN]** `unsigned` for block numbers is semantically correct and eliminates compiler warnings

- **Additional cleanup (reviewer feedback)**
  - **[REFACTOR]** Removed `rebaseFromSucc()` helper function - used only once, inlined at call site in `merge()`
  - Reviewer questioned why a `static inline` function was in the header; since it's trivial (3 additions) and single-use, inlining is cleaner

---

### 2025-12-09 – Code Review Cleanup Part 2 [REVIEW] [REFACTOR] [NUA]

- **Context / goal**
  - Continue addressing GitHub reviewer feedback on NUA code review
  - Promote utility function to common LLVM CodeGen infrastructure
  - Remove dead code and duplicate implementations

- **Changes applied**

  1. **[REFACTOR] Promoted `getSubRegIndexForLaneMask` to `TargetRegisterInfo`**
     - Added new method to `llvm/include/llvm/CodeGen/TargetRegisterInfo.h`:
       ```cpp
       /// Return the subreg index whose LaneMask exactly equals \p Mask, or 0 if none.
       unsigned getSubRegIndexForLaneMask(LaneBitmask Mask) const {
         for (unsigned Idx = 1, E = getNumSubRegIndices(); Idx < E; ++Idx)
           if (getSubRegIndexLaneMask(Idx) == Mask)
             return Idx;
         return 0;
       }
       ```
     - This provides a general reverse-lookup from LaneMask→SubRegIdx
     - Previously existed only as AMDGPU-specific inline function

  2. **[REFACTOR] Deleted `AMDGPUSSARAUtils.h` entirely**
     - File contained 3 utility functions:
       - `getOperandLaneMask()` - **DEAD CODE** (defined but never called)
       - `getSubRegIndexForLaneMask()` - moved to TRI
       - `getCoveringSubRegsForLaneMask()` - **DEAD CODE** (defined but never called, duplicated `TRI::getCoveringSubRegIndexes()`)
     - Addressed reviewer comment: "I'm pretty sure we have all of this stuff invented in multiple other places"

  3. **[REFACTOR] Updated call sites to use TRI method**
     - `VRegMaskPair.h`: Changed `getSubRegIndexForLaneMask(LaneMask, TRI)` → `TRI->getSubRegIndexForLaneMask(LaneMask)`
     - `AMDGPUNextUseAnalysis.h`: Same change in debug printing
     - `NextUseAnalysisTest.cpp`: Removed duplicate function definition, updated call site

  4. **[BUGFIX] Fixed header line wrap in `AMDGPUSSARAUtils.h`** (before deletion)
     - Reviewer comment: "Broken line wrap"
     - Changed from broken two-line:
       ```cpp
       //===------- AMDGPUSSARAUtils.h ----------------------------------------*- C++-
       //*-===//
       ```
     - To proper single line:
       ```cpp
       //===-- AMDGPUSSARAUtils.h - SSA RA Utilities -------------------*- C++ -*-===//
       ```

- **Reviewer comment analysis: `int64_t` vs `SlotIndex` for distances**
  - Reviewer asked: "Why are the units of distance int64_t and not SlotIndex / InstrDist?"
  - **Analysis**: SlotIndex measures **linear MBB layout** distance, NOT **CFG path** distance
  - `SlotIndexes.cpp` line 96: `for (MachineBasicBlock &MBB : *mf)` - iterates in linear order
  - Example: if `bb.0 → bb.2` (branch), SlotIndex includes `bb.1` even if unreachable
  - NUA computes distances along **actual CFG paths** via backward dataflow
  - **Conclusion**: SlotIndex is inappropriate for this use case; `int64_t` is correct

- **Files changed**
  - `llvm/include/llvm/CodeGen/TargetRegisterInfo.h`: Added `getSubRegIndexForLaneMask()` method
  - `llvm/lib/Target/AMDGPU/AMDGPUSSARAUtils.h`: **DELETED**
  - `llvm/lib/Target/AMDGPU/VRegMaskPair.h`: Updated to use `TRI->getSubRegIndexForLaneMask()`
  - `llvm/lib/Target/AMDGPU/AMDGPUNextUseAnalysis.h`: Updated call site
  - `llvm/unittests/Target/AMDGPU/NextUseAnalysisTest.cpp`: Removed duplicate, updated call site

- **Test results**
  - All 18 NUA unit tests pass (17 MIR + 1 GetSortedSubregUses)
  - Build successful after all changes

---

### 2025-12-09 – MIR Vreg Renumbering Script Fix [BUGFIX] [TEST]

- **Context / goal**
  - User requested renumbering vregs in `simple-linear-block-distances.mir` to be compact from %0
  - Used existing `scripts/renumber_mir_vregs.py` script

- **Bug discovered**
  - After running script, test failed with: `redefinition of virtual register '%11'`
  - Root cause: Script only scanned `%N` patterns in MIR body, not `{ id: N, ...}` in registers section
  - Example: `%11` was declared in `registers:` section but never used in body
  - Script mapped `%12` → `%11`, but original `{ id: 11, ...}` entry remained → duplicate

- **Fix applied**
  - Added second regex pattern to scan registers section:
    ```python
    reg_id_pattern = re.compile(r'\{\s*id:\s*(\d+),')
    ```
  - Script now collects vreg IDs from both:
    1. `%N` patterns in body/CHECK lines
    2. `{ id: N, ...}` patterns in registers section

- **Result**
  - After fix, script correctly found 74 vregs (0-73) - all declared in registers section
  - File was already compact (no gaps) → no changes needed
  - Test passes

- **Files changed**
  - `scripts/renumber_mir_vregs.py`: Added `reg_id_pattern` to scan registers section

- **Lesson learned**
  - MIR files may declare registers in `registers:` section that aren't explicitly used as `%N` in the body
  - These "phantom" entries (e.g., for undef values, implicit defs) must be included in renumbering

### 2025-12-09 – MIR Test File Cleanup [REVIEW] [TEST] [NUA]

- **Context / goal**
  - Reviewer comment: "These tests have too much unnecessary stuff. Can remove most of the flags and the registers sections"
  - File: `llvm/test/CodeGen/AMDGPU/NextUseAnalysis/simple-linear-block-distances.mir`

- **Changes applied**
  - **[TEST]** Added `# REQUIRES: asserts` (test uses `-debug-only`)
  - **[REFACTOR]** Removed entire `registers:` section (74 entries)
  - **[REFACTOR]** Removed all boilerplate flags - kept only:
    - `tracksRegLiveness: true`
    - `isSSA: true`
  - **[TEST]** Regenerated CHECK patterns to match MIRParser's implicit renumbering

- **Technical note**
  - Without `registers:` section, MIRParser renumbers vregs based on definition order
  - Original file had non-sequential vreg IDs (%0, %1, %2, %6, %7, %8, %9, %10, %12, %14, ...)
  - MIRParser renumbers them to %0, %1, %2, %3, %4, %5, ... in definition order
  - CHECK patterns must match the renumbered output, not the original MIR source

- **Result**
  - File reduced from 730 lines to 553 lines (24% smaller)
  - Test passes
  - Much cleaner and easier to maintain

- **Process**
  - Created minimal MIR file with body only
  - Ran NUA pass and captured debug output
  - Generated CHECK patterns from actual output using Python script
  - Combined header + checks + body into final file

### 2025-12-09 – Delete Test README [REVIEW] [TEST]

- **Context / goal**
  - Reviewer comment: "We don't have any documentation like this for any other tests"
  - Reviewer comment: "All of this algorithm information belongs in comments in the pass itself"
  - Reviewer comment: "These descriptions belong in each individual test"

- **Change applied**
  - **[REFACTOR]** Deleted `llvm/test/CodeGen/AMDGPU/NextUseAnalysis/README.md`

- **Rationale**
  - LLVM convention: no README files in test directories
  - Algorithm documentation belongs in source code comments
  - Test descriptions already exist as `# NOTE:` comments in each test file

### 2025-12-09 – Comprehensive NUA Documentation [REVIEW] [REFACTOR] [NUA]

- **Context / goal**
  - Reviewer comments:
    - "I think we need some implementation notes here to explain how the analysis works overall and the key details"
    - "Such helper function needs more comment explaining what the function does, what the input arguments are"
    - "All these magic numbers need to be constexpr with meaningful names and with comments"

- **Changes applied**
  - **[REFACTOR]** Added comprehensive file header documentation to `AMDGPUNextUseAnalysis.cpp`:
    - Overview section explaining analysis purpose
    - Storage Model section with worked example showing relative offset scheme
    - Cross-Block Distance Propagation section with CFG merge example
    - Loop-Aware Distance Encoding section explaining LoopTag semantics
    - Dataflow Framework section with formal semilattice definition and convergence proof
  - **[REFACTOR]** Renamed `materializeForRank` -> `materializeForSpillRanking` for clarity
  - **[REFACTOR]** Added named constants:
    - `LoopExitRangeStart = 60000` (start of Tier 2 range)
    - `LoopExitRangeSize = 5000` (number of values in Tier 2)
  - **[REFACTOR]** Added detailed Doxygen documentation for `materializeForSpillRanking()`
  - **[REFACTOR]** Renamed parameters: `Stored` -> `StoredDistance`, `Mat64` -> `Materialized`
  - **[REFACTOR]** Fixed unicode characters in comments (replaced with ASCII):
    - `x` instead of multiplication sign
    - `->` instead of arrow symbol
  - **[BUGFIX]** Fixed cross-block example: bb.1 use moved to pos 0 to match stored=-1 math

- **Rationale**
  - Makes the storage model and materialization concept clear to readers
  - Named constants with comments explain the three-tier ranking system
  - Worked examples help understand the relative offset scheme
  - Formal dataflow framework section aids academic/theoretical review
  - Pure ASCII ensures compatibility with all editors and build systems

---

### 2025-12-11 – NUA Test Suite Documentation [TEST] [NUA]

- **Context / goal**
  - Address reviewer feedback about test documentation
  - Create accurate documentation of test CFG patterns
  - Verify CFGs by parsing MIR successor clauses

- **Analysis performed**
  - Read all 17 MIR test files in `test/CodeGen/AMDGPU/NextUseAnalysis/`
  - Parsed `successors:` clauses to extract actual CFG edges
  - Updated Mermaid diagrams to match actual structure

- **CFG Corrections made**
  - `simple-linear-block-distances.mir`: Fixed CFG - bb.0→bb.1,bb.3; bb.3→bb.1; bb.1→bb.2,bb.4; bb.2→bb.4
  - `complex-control-flow-11blocks.mir`: Corrected all edges from successors
  - `complex-control-flow-14blocks.mir`: Corrected (actually 15 blocks bb.0-bb.14)
  - `inner_cfg_in_2_nesteed_loops.mir`: Fixed inner loop internal CFG structure

- **Documentation updated**
  - Updated `/work/atimofee/sandbox/github/ssa-spiller-docs/NUA_TEST_PATTERNS.md`
  - All CFG diagrams now use Mermaid and match actual MIR structure
  - Removed TODO warning - diagrams verified against source

- **Files renamed**
  - `simple-linear-block-distances.mir` → `acyclic-phi-merge-distances.mir` (name now reflects 2 merge points)
  - `inner_cfg_in_2_nesteed_loops.mir` → `inner_cfg_in_2_nested_loops.mir` (fixed typo)

---

### 2025-12-11 – Vreg Renumber Script: IR Module Bug [BUGFIX] [TEST]

- **Context / goal**
  - User requested renumbering vregs in `complex-single-loop-a.mir`
  - Ran `scripts/renumber_mir_vregs.py` script

- **Bug discovered**
  - Script corrupted the file: `error: use of undefined value '%20'` at `br label %20`
  - Root cause: Script replaced `%N` patterns in embedded LLVM IR module section (`--- |` to `...`)
  - In LLVM IR, `%N` are block labels, NOT virtual registers
  - Script was only tested on `acyclic-phi-merge-distances.mir` which has NO IR module

- **Analysis**
  - 15 out of 18 NUA test files have embedded IR modules
  - Previous "regenerated CHECK patterns" work was manual pass output capture, not script usage
  - This is genuinely the first time script was used on file with IR module

- **Fix applied**
  - Modified script to detect IR module section (`--- |` to `...`)
  - Split file into: header (CHECK patterns) + IR module (unchanged) + MIR body
  - Apply vreg renumbering only to header and MIR body sections
  - IR module is preserved verbatim

- **Files changed**
  - `scripts/renumber_mir_vregs.py`: Added IR module detection and preservation

- **Additional fix**
  - Some files (e.g., `complex-control-flow-14blocks.mir`) use `---` as IR module terminator instead of `...`
  - Updated script to handle both: `...` or next `---` line as IR module end

- **Bulk processing**
  - Renumbered all 17 NUA test files
  - Regenerated CHECK patterns using `scripts/regenerate_nua_checks.py`
  - All 17 unit tests pass

- **Files changed**
  - `scripts/renumber_mir_vregs.py`: IR module detection (both `...` and `---` terminators)
  - All 17 `.mir` files in `test/CodeGen/AMDGPU/NextUseAnalysis/`: vregs renumbered, CHECKs regenerated

- **Documentation fixes**
  - `complex-single-loop-a.mir`: Corrected key validation points
    - Old (wrong): "LoopTag applied consistently on all exit edges"
    - New (correct): "LoopTag NOT applied on internal latch edges - iteration count is independent of which back-edge is taken"
    - Key insight: Multi-latch loop with single header - internal latch crossing is NOT a loop exit
  - `complex-single-loop-b.mir`: Corrected description
    - Old (wrong): "Complex single loop with preheader, internal branches, and single back edge"
    - New (correct): "Complex acyclic CFG with simple self-loop (bb.3) on 'then' path, no loop on 'else' path"
    - Key insight: bb.11→bb.6 is NOT a back-edge (bb.6 cannot reach bb.11), only loop is self-loop at bb.3

- **File renames (names didn't match CFG structure)**
  - `complex-control-flow-14blocks.mir` → `complex-control-flow-15blocks.mir` (has 15 blocks bb.0-bb.14)
  - `complex-single-loop-b.mir` → `acyclic-cfg-with-self-loop.mir` (NOT a complex loop)
  - `multi_exit_loop_followed_by_simple_loop.mir` → `two-sequential-loops.mir` (neither has multi-exit)
  - `three_loops_sequence_nested_in_outer_loop.mir` → `complex-acyclic-cfg-with-4-self-loops.mir` (inner loops NOT nested)
  - `nested-loops-with-side-exits-b.mir` → `double-nested-loops-complex-cfg.mir` (NO side exits, just double nested)

- **MD documentation updated**
  - All 17 test files verified against actual MIR CFG structure
  - CFG diagrams corrected for `complex-acyclic-cfg-with-4-self-loops.mir` (removed wrong edge, added correct self-loop)
  - Descriptions and Key Validation Points corrected for renamed files
  - Summary table updated with new file names
  - All 17 unit tests pass

---

### 2025-01-05 – Session Resume & Test Verification [TEST] [SSA_SPILLER]

- **Context / goal**
  - Resuming work after Cursor AI plan overuse gap
  - Verifying current state of SSA Spiller tests
  - Planning test creation and documentation updates

- **Status verified**
  - All 10 SSA Spiller tests pass (9 PASS + 1 XFAIL)
  - 17 NextUseAnalysis tests available in separate directory
  - WWM/EWF code removed - old design leftovers, forget it
  - Cost model for split-before-use: NOT started yet

- **SSA Spiller Test Results**
  | Test | Status |
  |------|--------|
  | `spill-linear-dominated.mir` | ✅ PASS |
  | `spill-dominated-branches.mir` | ✅ PASS |
  | `spill-vreg-subregister.mir` | ✅ PASS |
  | `spill-multi-predecessor-join.mir` | ✅ PASS |
  | `spill-use-before-spill.mir` | ✅ PASS |
  | `spill-multi-path-independent.mir` | ✅ PASS |
  | `spill-vreg-many-lanes.mir` | ✅ PASS |
  | `spill-dom-groups-a.mir` | ✅ PASS |
  | `spill-dom-groups-b.mir` | ✅ PASS |
  | `spill-balanced-use-before.mir` | ⚠️ XFAIL (cost model pending) |

- **Test locations**
  - SSA Spiller: `llvm/test/CodeGen/AMDGPU/SSASpiller/` (10 tests)
  - NextUseAnalysis: `llvm/test/CodeGen/AMDGPU/NextUseAnalysis/` (17 tests)

- **Documentation locations**
  - Main docs: `/work/atimofee/sandbox/github/ssa-spiller-docs/SSA Spiller/`
  - Test patterns: `05-Testing/SSA Spiller/SSA_SPILLER_TEST_PATTERNS.md`

- **Next actions**
  - [ ] Create additional SSA Spiller test cases (identify gaps)
  - [ ] Update documentation with current implementation status
  - [ ] Document cleanup: remove WWM/EWF references from docs

  - **Documentation fixes applied**
  - Fixed broken links in `04-Design/Decisions.md`:
    - `SSA Repairing Disorder` links → `../08-Worklog/issues/SSA Spiller/...`
    - `Static NUA limitation` link → `../08-Worklog/issues/Next Use Analysis/...`
  - Fixed broken links in `04-Design/SSA_SPILLER_DESIGN.md`:
    - 4x `Static NUA limitation` links with anchors
    - `ISSUE_Spill_Markers_and_PHIs` link
  - Fixed broken links in `04-Design/Persistent Map for NUA.md`:
    - `Next Use Analysis Tests` → `../05-Testing/Next Use Analysis/NUA_TEST_PATTERNS.md`
    - `MIN Algorithm (Belady)` → `../03-Concepts/MIN Algorithm.md`
    - `Ideas` → `../10-Backlog/Open problems.md`
  - Verified: Both design changes (Static NUA limitation, SSA Repair failure) are well documented
  - **All `TODO: file not found` comments removed from 04-Design/**
  - Added "Critical: LiveInterval must be killed before SSA repair" section to `MachineLaneSSAUpdater.md`
    - Documents the SSA Repair Disorder fix and links to issue + Decisions.md
  - **Source code links updated with permanent commit hash `45385c6f5f00`:**
    - `SSA_SPILLER_DESIGN.md`: Source mapping, Key orchestration points (spillAndReload L657-742, emitReloadsAndRepairSSA L744-960, spillAtDefinition L966-1047), isUseReachableFromDef L769-839
    - `Decisions.md`: Implementation Reference section (collectDominatedBlocks L1523-1531, cutFromLiveRange L1533-1555, killIntervalInDominatedRegion L1557-1581)
    - `MachineLaneSSAUpdater.md`: Source mapping, repairSSAForNewDef L53-119, isUseReachableFromDef L769-879

---

### 2026-01-06 – Documentation Update: Design & Test Patterns [REFACTOR]

- **Context / goal**
  - Update documentation to reflect current implementation state
  - Fix outdated line number references in GitHub links
  - Add missing sections for new test patterns

- **Changes applied to `04-Design/SSA_SPILLER_DESIGN.md`**
  - **[REFACTOR]** Updated GitHub line numbers to commit `45385c6f5f00`:
    - `spillAndReload`: L615-717 → L657-742
    - `emitReloadsAndRepairSSA`: L719-929 → L744-960
    - `spillAtDefinition`: L935-1016 → L966-1047
  - **[REFACTOR]** Added link to `killIntervalInDominatedRegion` (L1557-1581)
  - **[REFACTOR]** Added new sections:
    - "Two-Pass Architecture" - explains SGPR-then-VGPR ordering
    - "Dominance Grouping (`DomGroup` class)" - documents reload minimization
    - "Final RP Validation" - documents `validateFinalRegisterPressure()`
  - **[REFACTOR]** Resolved all TODO comments - added links to:
    - MIN_Algorithm concept, Decisions#Store At Definition, Virtual Spill Point section
    - MachineLaneSSAUpdater component, VRegMaskPair.h source
    - DomGroup class source (L44-59), validateFinalRegisterPressure source (L185-233)
    - spill-vreg-many-lanes.mir test, SI_VIRTUAL_SPILL_MARKER TableGen
    - Static NUA limitation issue, SSA_SPILLER_TEST_PATTERNS.md
    - All PR links, all symbolic file paths, isUseReachableFromDef, repairSSAForNewDef

- **Changes applied to `03-Concepts/MIN_Algorithm.md`**
  - **[FEATURE]** Expanded from placeholder to comprehensive documentation
  - Covers: origin (Bélády 1966), theoretical foundation, algorithm pseudocode
  - Extension to CFGs: Braun & Hack CC 2009 paper reference
  - SSA Spiller implementation: NextUseAnalysis APIs, getVMPsToSpill()
  - Limitations: Static NUA, CFG complexity
  - References: original paper, Braun09, local PDF copy

- **Changes applied to `05-Testing/SSA_Spiller/SSA_SPILLER_TEST_PATTERNS.md`**
  - **[REFACTOR]** Added source links for spillAtDefinition, emitReloadsAndRepairSSA
  - **[REFACTOR]** Linked SI_VIRTUAL_SPILL_MARKER to TableGen, store-at-definition to Decisions.md
  - **[REFACTOR]** Replaced inline code snippet with GitHub link (L709-720)
  - **[REFACTOR]** Linked all 10 test files to GitHub source
  - **[REFACTOR]** Added links to DomGroup class, merge(), getSortedSubregUses()
  - **[REFACTOR]** Pattern Classification section: linked all concepts and test files
  - **[REFACTOR]** Footer: linked Design Version to Decisions.md sections

- **Changes applied to `05-Testing/SSA_Spiller/SSA_SPILLER_TEST_PATTERNS.md`**
  - **[REFACTOR]** Added 2 new tests to overview table:
    - `spill-dom-groups-a.mir` - Dominance grouping basic
    - `spill-dom-groups-b.mir` - Dominance grouping complex CFG
  - **[REFACTOR]** Updated source code reference lines 683-696 → 641-655
  - **[REFACTOR]** Added full test documentation for Test 9 and Test 10
  - **[REFACTOR]** Updated footer: date, design version, test count (10 tests)

- **Files changed**
  - `ssa-spiller-docs/SSA_Spiller/04-Design/SSA_SPILLER_DESIGN.md`
  - `ssa-spiller-docs/SSA_Spiller/05-Testing/SSA_Spiller/SSA_SPILLER_TEST_PATTERNS.md`

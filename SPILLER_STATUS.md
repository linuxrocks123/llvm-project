## SSA Spiller Current Status (2025-11-29)

### Active TODOs
1. **Regenerate SSASpiller CHECKs (skip balanced-use, fix gfx900)**  
   - Re-run `update_mir_test_checks.py` for every MIR test except `spill-balanced-use-before.mir`.  
   - Mark `spill-balanced-use-before.mir` as `XFAIL` (balanced spills still unimplemented).  
   - Ensure `spill-vreg-many-lanes.mir` keeps `-mcpu=gfx900` in its `RUN:` line after regenerating checks.
2. **Refactor `hasUseOnPath()` to use NextUseAnalysis data**  
   - Replace the custom BFS that scans every instruction with calls to `NextUseResult::usedInBlock(MBB)` while walking the CFG.  
   - Goal: reuse cached lane/usage info and cut compile time for spill placement decisions.
3. **Classify reachable uses inline**  
   - Currently we collect `NonDominatedUses` (lines 794-797) and then re-iterate (lines 806-823) to query `SSAUpdater::isUseReachableFromDef`.  
   - Fold the reachability test into the first loop so we only touch each use once.

### Recent History / Context
- **Register-pressure limit fix:** the 90% safety margin had been commented out temporarily (lines 1736-1738). It’s now restored so tests actually trigger spills again.
- **Test suite state:**  
  - All MIR tests run with `--amdgpu-ssa-spill-markers=1`; `spill-balanced-use-before.mir` still represents an unimplemented feature.  
  - `spill-vreg-many-lanes.mir` must keep `-mcpu=gfx900`; running markers on it previously hit verifier issues when limits were wrong.
- **Performance investigations:**  
  - `hasUseOnPath()` (lines 1421-1458) is a compile-time hotspot; leveraging `NextUseAnalysis` should fix that.  
  - Reachable-use classification has redundant loops that we plan to merge (Task 3).

### Files of Interest
- `llvm/lib/Target/AMDGPU/AMDGPUSSARegisterSpiller.cpp`
- `llvm/lib/Target/AMDGPU/AMDGPUNextUseAnalysis.{h,cpp}`
- `llvm/test/CodeGen/AMDGPU/SSASpiller/*.mir`

This summary can be pasted into any chat to restore context quickly.



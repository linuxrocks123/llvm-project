; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global \
; RUN:   -amdgpu-atomic-optimizer-strategy=Iterative -amdgpu-ssa-regalloc \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
;
; Regression test for a MachineLaneSSAUpdater bug exposed via AMDGPURebuildSSA.
; A signed i64 "varying" atomic reduction lowers to a ComputeLoop whose 64-bit
; accumulator is loop-carried with per-lane (sub0/sub1) defs. RebuildSSA rebuilds
; SSA by renaming the lane defs and inserting lane-aware loop-header PHIs. The
; updater used to fill a loop-header PHI's preheader operand with an
; OrigVReg.subIdx placeholder, relying on a later rewrite to patch it. For a lane
; whose preheader def was renamed *before* the loop PHI was created (a non-Root
; lane), the placeholder was never resolved, leaving a PHI operand with no
; reaching def -> "Reading virtual register without a def" / verifier abort.
;
; The fix resolves preheader operands to the actual reaching def (the renamed
; lane vreg live-out of the predecessor). Regression = this compiles and passes
; -verify-machineinstrs.

@local_var32 = external addrspace(3) global i32, align 4

define amdgpu_kernel void @max_i64_varying(ptr addrspace(1) %out) {
entry:
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %lane_ext = zext i32 %lane to i64
  %old = atomicrmw max ptr addrspace(3) @local_var32, i64 %lane_ext acq_rel, align 8
  store i64 %old, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

; CHECK-LABEL: max_i64_varying:
; The signed-i64 max reduction loop must be emitted (compare + per-lane select).
; CHECK: v_cmp_gt_i64
; CHECK: s_cselect_b32
; CHECK: s_cselect_b32
; CHECK: s_cbranch
; Reduced i64 result stored back; no spilling.
; CHECK: buffer_store_dwordx2
; CHECK: ScratchSize: 0

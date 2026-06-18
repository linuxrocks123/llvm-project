; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s
;
; End-to-end SSA RA pipeline test (T2c): loop carrying divergent loop-invariant
; values across the body under a tight VGPR budget (loop + back-edge + spill).
;
; XFAIL: *
; REQUIRES: asserts
; Known failure: SSA RA color() aborts ("Failed to find free physreg") on a
; budget greedy fits without spilling. RebuildSSA legalizes in-place subregister
; defs into a partial wide def (undef %r.sub0:vreg_64) that costs a full tuple
; for a single lane, inflating loop pressure past the budget (see NOTES
; 2026-06-18). Fixed by the MachineLaneSSAUpdater refactor (Phase 1); flip this
; to a real CHECK once that lands.

define amdgpu_kernel void @pipeline_spill_loop(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %base = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %b0 = load volatile i32, ptr addrspace(1) %base
  %q1 = getelementptr i32, ptr addrspace(1) %base, i32 1
  %b1 = load volatile i32, ptr addrspace(1) %q1
  %q2 = getelementptr i32, ptr addrspace(1) %base, i32 2
  %b2 = load volatile i32, ptr addrspace(1) %q2
  %q3 = getelementptr i32, ptr addrspace(1) %base, i32 3
  %b3 = load volatile i32, ptr addrspace(1) %q3
  br label %loop

loop:
  %i   = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %loop ]
  %p   = getelementptr i32, ptr addrspace(1) %base, i32 %i
  %v   = load volatile i32, ptr addrspace(1) %p
  %t0  = add i32 %v, %b0
  %t1  = add i32 %t0, %b1
  %t2  = add i32 %t1, %b2
  %t3  = add i32 %t2, %b3
  %acc.next = add i32 %acc, %t3
  %i.next = add i32 %i, 1
  %cmp = icmp ult i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  store i32 %acc.next, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
attributes #0 = { "amdgpu-num-vgpr"="6" }

; CHECK-LABEL: pipeline_spill_loop:
; CHECK: global_store_dword

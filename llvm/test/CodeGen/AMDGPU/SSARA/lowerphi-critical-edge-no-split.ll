; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global \
; RUN:   -amdgpu-atomic-optimizer-strategy=Iterative -amdgpu-ssa-regalloc \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
;
; SSA-destruction PHI lowering (AMDGPUSSARegisterAllocator): a uniform i64 atomic
; carries its 64-bit accumulator via two per-lane PHIs on a critical (triangle)
; edge. After coloring, the PHI-result colors and their sources form a
; VGPR0<->VGPR1 permutation cycle whose registers are dead on the predecessor's
; other out-edge, so the copies are placed at the predecessor's terminator and
; the cycle is resolved in place with v_swap_b32 -- the critical edge is NOT
; split. (lowerPHIs used to split every critical edge unconditionally, which drove
; SplitCriticalEdge down a live-interval update path that asserted.)

define amdgpu_kernel void @add_i64_uniform(ptr addrspace(1) %out, ptr addrspace(1) %inout, i64 %additive) {
entry:
  %old = atomicrmw add ptr addrspace(1) %inout, i64 %additive syncscope("agent") acq_rel, align 8
  store i64 %old, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: add_i64_uniform:
; Per-lane accumulator swap resolved in place -- no split block, no scratch mov.
; CHECK: v_swap_b32 v1, v0
; CHECK: buffer_atomic_add_x2 v[0:1]
; CHECK: buffer_store_dwordx2 v[0:1]
; CHECK: s_endpgm

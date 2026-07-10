; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global \
; RUN:   -amdgpu-atomic-optimizer-strategy=Iterative -amdgpu-ssa-regalloc \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
;
; SSA-destruction PHI lowering (AMDGPUSSARegisterAllocator): a uniform i64 atomic
; carries its 64-bit accumulator via two per-lane PHIs on a critical (triangle)
; edge. The critical edge must NOT be split during PHI lowering. (lowerPHIs used
; to split every critical edge unconditionally, which drove SplitCriticalEdge down
; a live-interval update path that asserted.) Here coloring assigns the per-lane
; PHI results the same registers as their sources (the atomic result in v[0:1]),
; so SSA destruction inserts no copies at all and the edge stays unsplit.

define amdgpu_kernel void @add_i64_uniform(ptr addrspace(1) %out, ptr addrspace(1) %inout, i64 %additive) {
entry:
  %old = atomicrmw add ptr addrspace(1) %inout, i64 %additive syncscope("agent") acq_rel, align 8
  store i64 %old, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: add_i64_uniform:
; The critical (triangle) edge is NOT split (guarded by -verify-machineinstrs not
; asserting). Coloring aligns the per-lane accumulator PHI results with the atomic
; result in v[0:1], so SSA destruction needs no copy at all -- no split block, no
; scratch mov, no swap.
; CHECK: buffer_atomic_add_x2 v[0:1]
; CHECK: s_or_b64 exec, exec,
; CHECK: buffer_store_dwordx2 v[0:1]
; CHECK: s_endpgm

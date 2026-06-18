; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s
;
; End-to-end SSA RA pipeline test (T3c): 128-bit vector load/store (vreg_128).
; Verifies the 4-wide store is emitted correctly through the full SSA RA pipeline.

define amdgpu_kernel void @pipeline_wide_v128(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p = getelementptr <4 x i32>, ptr addrspace(1) %in, i32 %tid
  %a = load <4 x i32>, ptr addrspace(1) %p
  %b = add <4 x i32> %a, <i32 1, i32 2, i32 3, i32 4>
  %o = getelementptr <4 x i32>, ptr addrspace(1) %out, i32 %tid
  store <4 x i32> %b, ptr addrspace(1) %o
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

; CHECK-LABEL: pipeline_wide_v128:
; CHECK: global_store_dwordx4

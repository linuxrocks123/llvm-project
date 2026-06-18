; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s
;
; End-to-end SSA RA pipeline test (T3b): 64-bit element load/store exercises
; vreg_64 tuple allocation.

define amdgpu_kernel void @pipeline_wide_v64(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p = getelementptr i64, ptr addrspace(1) %in, i32 %tid
  %a = load i64, ptr addrspace(1) %p
  %b = add i64 %a, 17
  %o = getelementptr i64, ptr addrspace(1) %out, i32 %tid
  store i64 %b, ptr addrspace(1) %o
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

; CHECK-LABEL: pipeline_wide_v64:
; CHECK: global_load_dwordx2 v[{{[0-9:]+}}],
; CHECK: global_store_dwordx2

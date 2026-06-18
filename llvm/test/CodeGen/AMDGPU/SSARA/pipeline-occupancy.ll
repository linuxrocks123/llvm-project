; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s
;
; End-to-end SSA RA pipeline test (T4b): a low-pressure kernel under an
; amdgpu-num-vgpr=32 budget uses only a few VGPRs and does not spill.

define amdgpu_kernel void @pipeline_occupancy(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %a = load i32, ptr addrspace(1) %p
  %b = add i32 %a, %tid
  %o = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %b, ptr addrspace(1) %o
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
attributes #0 = { "amdgpu-num-vgpr"="32" }

; CHECK-LABEL: pipeline_occupancy:
; CHECK: global_load_dword
; CHECK: global_store_dword
; CHECK: .set pipeline_occupancy.num_vgpr, 3

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s
;
; End-to-end SSA RA pipeline test (T4a): all-uniform values stay scalar (SGPR);
; the SSA RA must not inflate them into VGPRs.

define amdgpu_kernel void @pipeline_sgpr_heavy(ptr addrspace(1) %out, i32 %a, i32 %b, i32 %c, i32 %d) {
  %s0 = add i32 %a, %b
  %s1 = add i32 %s0, %c
  %s2 = add i32 %s1, %d
  %s3 = mul i32 %s2, %a
  %s4 = mul i32 %s3, %b
  store i32 %s4, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: pipeline_sgpr_heavy:
; CHECK: s_add_i32
; CHECK: s_mul_i32
; CHECK: global_store_dword
; CHECK: NumVgprs: 2

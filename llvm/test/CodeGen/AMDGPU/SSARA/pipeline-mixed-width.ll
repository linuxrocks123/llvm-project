; RUN: llc -mtriple=amdgcn -mcpu=gfx908 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s
;
; End-to-end SSA RA pipeline test (T3a): MFMA with a wide accumulator exercises
; mixed VGPR/AGPR allocation through the full pipeline.

define amdgpu_kernel void @pipeline_mixed_width(ptr addrspace(1) %out, float %a, float %b, <4 x float> %c) #0 {
  %r = call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0)
  store <4 x float> %r, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float, float, <4 x float>, i32, i32, i32)
attributes #0 = { "target-cpu"="gfx908" }

; CHECK-LABEL: pipeline_mixed_width:
; CHECK: v_mfma_f32_4x4x1f32 a[{{[0-9:]+}}], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9:]+}}]
; CHECK: global_store_dwordx4

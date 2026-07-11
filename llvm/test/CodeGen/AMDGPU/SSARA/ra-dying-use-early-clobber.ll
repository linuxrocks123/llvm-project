; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for not reusing a dying use's physreg for an early-clobber def
; (AMDGPUSSARegisterAllocator coloring).
;
; An MFMA accumulator (v_mfma_..., an early-clobber two-address def) reads a wide
; value that dies at the instruction. Coloring freed the dying use's physregs and
; then let the early-clobber def reuse them -- but an early-clobber def is written
; before the reads complete, so the reused register is read while undefined.
; -verify-machineinstrs aborted with "Using an undefined physical register". The
; fix colors the early-clobber def against the occupied set BEFORE dying uses are
; freed, so it does not reuse a register the instruction still reads.

; CHECK-LABEL: test_load_mfma_store16:
; CHECK: v_mfma_f32_32x32x1f32
; CHECK: s_endpgm
define amdgpu_kernel void @test_load_mfma_store16(ptr addrspace(1) %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %arg, i32 %tid
  %in.1 = load <32 x float>, ptr addrspace(1) %gep, align 128
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %in.1, i32 1, i32 2, i32 3)
  store <32 x float> %mai.1, ptr addrspace(1) %gep, align 128
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32 immarg, i32 immarg, i32 immarg)

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }

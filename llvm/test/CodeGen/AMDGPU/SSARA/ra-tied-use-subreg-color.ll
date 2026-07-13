; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for coloring a two-address def tied to a SUB-register of its use's value
; (AMDGPUSSARegisterAllocator::color).
;
; writelane of a <2 x float> lowers to V_WRITELANE_B32 whose def is tied to one
; 32-bit lane of a wider value. The def inherited the tied use's WHOLE
; super-register color, so the 32-bit lane def was assigned a vreg_64 physreg ->
; "Operand has incorrect register class". The fix inherits the sub-register of
; the tied use's color that the use actually reads.

; CHECK-LABEL: test_readlane_v2f32:
; CHECK: s_setpc_b64
define void @test_readlane_v2f32(ptr addrspace(1) %out, <2 x float> %src, i32 %src1) {
  %oldval = load <2 x float>, ptr addrspace(1) %out, align 8
  %writelane = call <2 x float> @llvm.amdgcn.writelane.v2f32(<2 x float> %src, i32 %src1, <2 x float> %oldval)
  store <2 x float> %writelane, ptr addrspace(1) %out, align 4
  ret void
}

declare <2 x float> @llvm.amdgcn.writelane.v2f32(<2 x float>, i32, <2 x float>)

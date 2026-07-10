; RUN: llc -global-isel -mtriple=amdgcn -mcpu=gfx1200 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for tied-use reaching-value queries in SSA reconstruction
; (MachineLaneSSAUpdater).
;
; A two-address WMMA accumulator is a read-modify-write with an early-clobber
; def. When RebuildSSA renames the WMMA's def to a fresh vreg, the WMMA's own
; tied use must be rewritten to read the incoming accumulator value. Querying
; the reaching value at the plain register slot resolved to the instruction's
; own early-clobber def, so the tied use was left dangling on the now-def-less
; OrigVReg. The fix queries a tied use at the instruction base index (the read
; point, before every def slot). Without it, -verify-machineinstrs aborted with
; "Reading virtual register without a def" and "Two-address instruction operands
; must be identical" on the WMMA tied-def operand.

; CHECK-LABEL: test_wmma_f32_16x16x16_f16:
; The accumulator (last operand) must read the same register range as the def,
; i.e. two-address form is preserved after reconstruction + coloring.
; CHECK: v_wmma_f32_16x16x16_f16 [[ACC:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], [[ACC]]
define amdgpu_ps void @test_wmma_f32_16x16x16_f16(<8 x half> %A, <8 x half> %B, <8 x float> %C, ptr addrspace(1) %out) {
bb:
  %res = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %A, <8 x half> %B, <8 x float> %C)
  store <8 x float> %res, ptr addrspace(1) %out, align 32
  ret void
}

declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>, <8 x half>, <8 x float>)

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for coloring a two-address def whose tied use is an undef passthrough
; (AMDGPUSSARegisterAllocator::color).
;
; A D16 "hi" load writes only the high 16 bits and ties the untouched low half
; as an undef passthrough use of the def's own vreg. That self-tied use has no
; earlier color to inherit, so the coloring assert "Tied use must be colored
; already" fired. The fix colors such a def via pickFreePhysReg (a free def);
; operand rewrite then assigns the same physreg to the self-tied use, preserving
; two-address form.

; CHECK-LABEL: load_local_hi_v2i16_undeflo:
; CHECK: ds_read_u16_d16_hi v0, v0
; CHECK: s_setpc_b64
define <2 x i16> @load_local_hi_v2i16_undeflo(ptr addrspace(3) %in) #0 {
entry:
  %load = load i16, ptr addrspace(3) %in, align 2
  %build = insertelement <2 x i16> poison, i16 %load, i32 1
  ret <2 x i16> %build
}

attributes #0 = { nounwind }

; RUN: llc -global-isel=1 -mtriple=amdgcn-pal -mcpu=gfx1010 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for physreg pressure accounting on undef uses in the SSA spiller
; (AMDGPUSSARegisterSpiller::processFunction).
;
; SI_RETURN_TO_EPILOG carries `implicit undef $vgprN` operands. The physreg
; pressure counter decremented for every dead-after use, including these undef
; uses, which were never counted -- underflowing the unsigned counter to a huge
; value. That forced a spurious spill, whose live-set walk then asserted on a
; classless GlobalISel-leftover vreg ("Register class not set"). The fix skips
; undef uses when releasing physreg pressure.

; CHECK-LABEL: _amdgpu_ps_1_arg:
define dllexport amdgpu_ps { <4 x float> } @_amdgpu_ps_1_arg(i32 inreg %arg, i32 inreg %arg1, i32 inreg %arg2, <2 x float> %arg3, <2 x float> %arg4, <2 x float> %arg5, <3 x float> %arg6, <2 x float> %arg7, <2 x float> %arg8, <2 x float> %arg9, float %arg10, float %arg11, float %arg12, float %arg13, float %arg14, i32 %arg15, i32 %arg16, i32 %arg17, i32 %arg18) {
.entry:
  %i1 = extractelement <2 x float> %arg3, i32 1
  %ret1 = insertelement <4 x float> poison, float %i1, i32 0
  %ret2 = insertvalue { <4 x float> } poison, <4 x float> %ret1, 0
  ret { <4 x float> } %ret2
}

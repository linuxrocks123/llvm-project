; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 -amdgpu-enable-rewrite-partial-reg-uses=false -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for rewriting a loop-carried PHI self-reference in SSA reconstruction
; (MachineLaneSSAUpdater::rewriteDominatedUses).
;
; A loop-header PHI that is the new def and reads OrigVReg on its own back-edge
; was skipped by the UseMI==DefMI guard, leaving that operand unrewritten with no
; reaching def. Here the <3 x float> loop-carried value has an undef subrange, so
; reconstruction splits it and the header PHI references itself across the back
; edge. Without the fix (skip the def instruction only when it is not a PHI),
; -verify-machineinstrs aborted with "Found PHI instruction with NoPHIs property
; set" and "Reading virtual register without a def".

; CHECK-LABEL: liveout_undef_subrange:
; CHECK: s_setpc_b64
define <3 x float> @liveout_undef_subrange(<3 x float> %arg) {
bb:
  br label %bb1

bb1:
  %i = phi <3 x float> [ %arg, %bb ], [ %i11, %bb3 ]
  %i2 = extractelement <3 x float> %i, i64 2
  %i3 = fmul float %i2, 1.000000e+00
  %i4 = fmul nsz <3 x float> %arg, splat (float 2.000000e+00)
  %i5 = insertelement <3 x float> poison, float %i3, i32 0
  %i6 = shufflevector <3 x float> %i5, <3 x float> poison, <3 x i32> zeroinitializer
  %i7 = fmul <3 x float> %i4, %i6
  %i8 = fcmp oeq float %i3, 0.000000e+00
  br i1 %i8, label %bb3, label %bb2

bb2:
  br label %bb3

bb3:
  %i11 = phi <3 x float> [ %i7, %bb2 ], [ %i, %bb1 ]
  br label %bb1
}

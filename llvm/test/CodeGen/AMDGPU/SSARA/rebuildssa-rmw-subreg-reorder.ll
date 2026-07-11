; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for per-block ordering of RMW subregister re-def renaming in SSA
; reconstruction (AMDGPURebuildSSA WorkList sort).
;
; A read-modify-write chain of partial subregister defs must rename the later
; (reader) defs before the establishing def they read. That reverse order was
; only applied when the whole vreg lived in a single block; a multi-block / loop
; vreg with a same-block RMW chain (as produced by the cross-block float PHIs
; here) was renamed in the wrong order, so RebuildSSA left a PHI in a block whose
; NoPHIs property was set. -verify-machineinstrs aborted with "Found PHI
; instruction with NoPHIs property set". The fix generalizes the reverse order to
; a same-block check in the WorkList sort.

; CHECK-LABEL: foo:
; CHECK: image_sample
; CHECK: s_endpgm
define amdgpu_ps void @foo() #0 {
bb:
  %undef0 = freeze i1 poison
  br i1 %undef0, label %bb2, label %bb1

bb1:
  %undef1 = freeze i1 poison
  br i1 %undef1, label %bb4, label %bb6

bb2:
  %tmp = phi float [ %tmp5, %bb4 ], [ 0.000000e+00, %bb ]
  br i1 poison, label %bb9, label %bb13

bb4:
  %tmp5 = phi float [ poison, %bb1 ], [ poison, %bb6 ], [ %tmp8, %bb7 ]
  br label %bb2

bb6:
  %undef2 = freeze i1 poison
  br i1 %undef2, label %bb7, label %bb4

bb7:
  %tmp8 = fmul float poison, poison
  br label %bb4

bb9:
  %tmp10 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32.v8i32.v4i32(i32 15, float poison, <8 x i32> poison, <4 x i32> poison, i1 false, i32 0, i32 0)
  %tmp11 = extractelement <4 x float> %tmp10, i32 1
  %tmp12 = extractelement <4 x float> %tmp10, i32 3
  br label %bb14

bb13:
  br i1 poison, label %bb23, label %bb24

bb14:
  %tmp15 = phi float [ %tmp12, %bb9 ], [ poison, %bb27 ], [ 0.000000e+00, %bb24 ]
  %tmp16 = phi float [ %tmp11, %bb9 ], [ poison, %bb27 ], [ %tmp25, %bb24 ]
  %tmp17 = fmul float 1.050000e+01, %tmp16
  %tmp18 = fmul float 1.150000e+01, %tmp15
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp18, float %tmp17, float %tmp17, float %tmp17, i1 true, i1 true)
  ret void

bb23:
  br i1 poison, label %bb24, label %bb26

bb24:
  %tmp25 = phi float [ %tmp, %bb13 ], [ %tmp, %bb26 ], [ 0.000000e+00, %bb23 ]
  br i1 poison, label %bb27, label %bb14

bb26:
  br label %bb24

bb27:
  br label %bb14
}

declare void @llvm.amdgcn.exp.f32(i32 immarg, i32 immarg, float, float, float, float, i1 immarg, i1 immarg)
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32.v8i32.v4i32(i32 immarg, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg)

attributes #0 = { nounwind }

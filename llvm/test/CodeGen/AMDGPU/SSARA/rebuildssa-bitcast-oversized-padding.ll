; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for lane-aware SSA reconstruction of a whole-register (super) use whose
; value lives in an oversized register class (MachineLaneSSAUpdater).
;
; A 384-/448-bit bitcast result merged by a whole-register PHI is held in an
; sgpr_512. LiveIntervalCalc fabricates a subrange over the never-defined
; padding lanes (sub12..sub15 / sub14..sub15) whose only values are undef inputs
; joined into a PHI-def. Reconstruction must (a) source those padding lanes as
; undef rather than a live OrigVReg placeholder that is never patched, and
; (b) split them into class-supported covering subregisters (sgpr_512 has no
; single sub12_..._sub15 / sub14_sub15 index). Without the fix,
; -verify-machineinstrs aborted with "Found PHI instruction with NoPHIs property
; set" and "Invalid subregister index for virtual register".

; CHECK-LABEL: bitcast_v12f32_to_v6f64_scalar:
; CHECK: s_setpc_b64
define inreg <6 x double> @bitcast_v12f32_to_v6f64_scalar(<12 x float> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <12 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <12 x float> %a1 to <6 x double>
  br label %end

cmp.false:
  %a3 = bitcast <12 x float> %a to <6 x double>
  br label %end

end:
  %phi = phi <6 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <6 x double> %phi
}

; CHECK-LABEL: bitcast_v14f32_to_v7f64_scalar:
; CHECK: s_setpc_b64
define inreg <7 x double> @bitcast_v14f32_to_v7f64_scalar(<14 x float> inreg %a, i32 inreg %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <14 x float> %a, splat (float 1.000000e+00)
  %a2 = bitcast <14 x float> %a1 to <7 x double>
  br label %end

cmp.false:
  %a3 = bitcast <14 x float> %a to <7 x double>
  br label %end

end:
  %phi = phi <7 x double> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <7 x double> %phi
}

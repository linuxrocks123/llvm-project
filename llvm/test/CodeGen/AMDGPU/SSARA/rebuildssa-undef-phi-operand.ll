; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for sourcing undef PHI operands as undef and lowering them to
; IMPLICIT_DEF (MachineLaneSSAUpdater::createPHIInBlockReaching +
; AMDGPUSSARegisterAllocator lowerPHIs).
;
; The whole-register PHI merging the two bitcast results has, per lane, an edge
; with no reaching def. createPHIInBlockReaching must source such an edge with
; the undef flag (a plain read is not live-out of the predecessor); lowerPHIs
; then materializes it as an IMPLICIT_DEF of the PHI-result physreg in the
; predecessor, like generic PHIElimination. Without the fix a live OrigVReg
; placeholder was emitted and verification aborted with "Found PHI instruction
; with NoPHIs property set".

; CHECK-LABEL: bitcast_v2f64_to_v8f16:
; CHECK: s_setpc_b64
define <8 x half> @bitcast_v2f64_to_v8f16(<2 x double> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = fadd <2 x double> %a, splat (double 1.000000e+00)
  %a2 = bitcast <2 x double> %a1 to <8 x half>
  br label %end

cmp.false:
  %a3 = bitcast <2 x double> %a to <8 x half>
  br label %end

end:
  %phi = phi <8 x half> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <8 x half> %phi
}

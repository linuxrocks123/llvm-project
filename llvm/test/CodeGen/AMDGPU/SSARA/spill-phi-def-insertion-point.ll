; RUN: llc -mtriple=amdgcn -mcpu=tahiti -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for the spill store insertion point when the spilled value is defined by
; a PHI (AMDGPUSSARegisterSpiller::spillAtDefinition).
;
; When a value defined by a PHI is spilled at its definition, the store must go
; after the LAST PHI in the block, not at std::next(PHI): all PHIs must stay
; contiguous at the block top. Here a wide (<40 x i8>) value merged by a PHI is
; spilled under register pressure; inserting the store at std::next(PHI) placed
; it between PHIs and -verify-machineinstrs aborted with "Found PHI instruction
; after non-PHI". The fix inserts the store at getFirstNonPHI().

; CHECK-LABEL: bitcast_v20i16_to_v40i8:
; CHECK: s_setpc_b64
define <40 x i8> @bitcast_v20i16_to_v40i8(<20 x i16> %a, i32 %b) {
  %cmp = icmp eq i32 %b, 0
  br i1 %cmp, label %cmp.true, label %cmp.false

cmp.true:
  %a1 = add <20 x i16> %a, splat (i16 3)
  %a2 = bitcast <20 x i16> %a1 to <40 x i8>
  br label %end

cmp.false:
  %a3 = bitcast <20 x i16> %a to <40 x i8>
  br label %end

end:
  %phi = phi <40 x i8> [ %a2, %cmp.true ], [ %a3, %cmp.false ]
  ret <40 x i8> %phi
}

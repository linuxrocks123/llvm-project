; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for finding the OrigVReg def among all operands of a re-def instruction
; in SSA reconstruction (MachineLaneSSAUpdater::repairSSAForNewDef).
;
; The re-def here is a variadic INLINEASM ("=v" vector output) whose def operand
; is not among the leading explicit defs -- MachineInstr::defs() returns them,
; and for a variadic INLINEASM that set is empty (its def operands sit after the
; asm string and flag immediates). Combined with the shuffle building a wider
; value, RebuildSSA splits the vreg and looks for its def; scanning only defs()
; found none. The fix scans all operands filtered by the isDef flag. Without it,
; RebuildSSA aborted: "NewDefMI should have a def operand for OrigVReg".

; CHECK-LABEL: v_shuffle_v3f32_v2f32__3_0_u:
; CHECK: global_store_dwordx3
; CHECK: s_setpc_b64
define void @v_shuffle_v3f32_v2f32__3_0_u(ptr addrspace(1) inreg %ptr) {
  %vec0 = call <2 x float> asm "; def $0", "=v"()
  %vec1 = call <2 x float> asm "; def $0", "=v"()
  %shuf = shufflevector <2 x float> %vec0, <2 x float> %vec1, <3 x i32> <i32 3, i32 0, i32 poison>
  store <3 x float> %shuf, ptr addrspace(1) %ptr, align 16
  ret void
}

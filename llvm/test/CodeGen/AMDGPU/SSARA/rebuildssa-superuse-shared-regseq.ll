; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for sharing one super-use REG_SEQUENCE across operands of the same
; instruction in SSA reconstruction (MachineLaneSSAUpdater::rewriteUseReaching).
;
; maximumnum lowers to V_MAX_F64 x, x (a self-canonicalize), reading the same
; 64-bit value in both operands. That value is built from two 32-bit partial
; subreg defs, so RebuildSSA reconstructs it via a super-use REG_SEQUENCE. Before
; the fix that REG_SEQUENCE was composed once per use operand, yielding two
; distinct vregs for one value -> two different registers feeding one VOP ->
; "VOP* instruction violates constant bus restriction". Sharing the composed
; value across the instruction's operands keeps the two reads identical.

; CHECK-LABEL: v_maximumnum_f64_s_v:
; CHECK: s_setpc_b64
define double @v_maximumnum_f64_s_v(double inreg %x, double %y) {
  %result = call double @llvm.maximumnum.f64(double %x, double %y)
  ret double %result
}

declare double @llvm.maximumnum.f64(double, double)

; RUN: llc -global-isel -amdgpu-codegenprepare-disable-idiv-expansion=1 -mtriple=amdgcn-mesa-mesa3d -mcpu=hawaii -denormal-fp-math-f32=preserve-sign -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for not leaking registers on dead defs in SSA-RA coloring
; (AMDGPUSSARegisterAllocator::color).
;
; A vector i64 division lowers to many V_ADD_CO_U32_e64 whose SGPR carry-out is
; dead (unused). color() colored each dead def and marked it occupied, but the
; kill path only frees dying uses -- never dead defs -- so they accumulated and
; exhausted SReg_64 ("Failed to find free physreg"). The fix skips marking a
; dead def occupied (it still gets a valid physreg and counts toward the
; high-water mark, but does not reserve a register going forward).

; CHECK-LABEL: v_sdiv_v2i64:
; CHECK: s_setpc_b64
define <2 x i64> @v_sdiv_v2i64(<2 x i64> %num, <2 x i64> %den) {
  %result = sdiv <2 x i64> %num, %den
  ret <2 x i64> %result
}

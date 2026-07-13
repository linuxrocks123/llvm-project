; RUN: llc -global-isel=0 -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for rewriting virtual operands of instructions nested inside BUNDLEs
; (AMDGPUSSARegisterAllocator::rewriteOperands).
;
; A GWS instruction is emitted as `BUNDLE implicit %r { DS_GWS_INIT %r, ... }`.
; Operand rewrite iterated the block top-level, which visits bundle headers but
; not the instructions inside bundles, so the DS_GWS_INIT operand was left as a
; virtual register -> "Remaining virtual register". The fix iterates instrs() so
; bundled instructions' operands are rewritten too.

; CHECK-LABEL: gws_init_offset0:
; CHECK: s_endpgm
define amdgpu_kernel void @gws_init_offset0(i32 %val) {
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 0)
  ret void
}

declare void @llvm.amdgcn.ds.gws.init(i32, i32)

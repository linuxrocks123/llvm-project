; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for rewriting an undef use tied to a def to the def's physreg
; (AMDGPUSSARegisterAllocator::rewriteOperands).
;
; PERMLANE reads its "old" passthrough source as an undef sub-register of a vreg
; that is never otherwise defined. It is tied to the def, so two-address form
; requires it to equal the def's physreg. The undef path picked an arbitrary
; allocatable physreg instead -> "Tied physical registers must match" /
; "Two-address instruction operands must be identical". The fix copies the tied
; def's already-rewritten physreg (dropping any sub-register).

; CHECK-LABEL: v_permlane16_b32_undef_tid_f64:
; CHECK: s_endpgm
define amdgpu_kernel void @v_permlane16_b32_undef_tid_f64(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %tidx_f32 = bitcast i32 %tidx to float
  %tidx_f64 = fpext float %tidx_f32 to double
  %undef = freeze double poison
  %v = call double @llvm.amdgcn.permlane16.f64(double %undef, double %tidx_f64, i32 %src1, i32 %src2, i1 false, i1 false)
  store double %v, ptr addrspace(1) %out, align 8
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare double @llvm.amdgcn.permlane16.f64(double, double, i32, i32, i1 immarg, i1 immarg)

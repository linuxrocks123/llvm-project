; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for coloring defs/uses by operand def/use flag rather than operand
; position (AMDGPUSSARegisterAllocator).
;
; A ds_wrxchg_rtn with an AGPR-constrained inline-asm value has a def operand
; that is not at a fixed position; a position-based scan mis-identified which
; operands are defs vs uses, leaving a real (non-undef) virtual register operand
; uncolored. Operand rewrite then aborted the assert
; "non-undef virtual register not colored". The fix classifies each operand by
; its isDef flag.

; CHECK-LABEL: ds_atomic_xchg_i32_ret_a_a:
; CHECK: ds_wrxchg_rtn_b32
; CHECK: s_setpc_b64
define void @ds_atomic_xchg_i32_ret_a_a(ptr addrspace(3) %ptr) #0 {
  %gep.0 = getelementptr inbounds [512 x i32], ptr addrspace(3) %ptr, i32 0, i32 10
  %data = call i32 asm "; def $0", "=a"()
  %result = atomicrmw xchg ptr addrspace(3) %ptr, i32 %data seq_cst, align 4
  call void asm "; use $0", "a"(i32 %result)
  ret void
}

attributes #0 = { nounwind "amdgpu-waves-per-eu"="10,10" }

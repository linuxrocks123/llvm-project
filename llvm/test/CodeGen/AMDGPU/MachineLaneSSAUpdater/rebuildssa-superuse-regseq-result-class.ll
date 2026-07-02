; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global \
; RUN:   -amdgpu-atomic-optimizer-strategy=Iterative -amdgpu-ssa-regalloc \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
;
; Regression test for a MachineLaneSSAUpdater bug exposed via AMDGPURebuildSSA
; (Family A: over-wide REG_SEQUENCE result class).
;
; When rewriting a super/mixed-lane use, buildRSForSuperUse materialized the
; needed lanes with a REG_SEQUENCE but sized the result to OrigVReg's *full*
; class (sgpr_128) instead of the use's class (the sub0_sub1 sreg_64), and
; cleared the operand subreg. The consumer (S_LOAD_DWORD_IMM base) then received
; a 128-bit vreg where a 64-bit one is required:
;
;   *** Bad machine code: Illegal virtual register for instruction ***
;   - instruction: %..:sreg_32_xm0_xexec = S_LOAD_DWORD_IMM %..:sgpr_128, 0, 0
;
; A uniform sub-32-bit (i16) atomicrmw is the smallest kernel that lowers to the
; unaligned-address path exercising this mixed super-use. The fix sizes the
; REG_SEQUENCE result to the use's register class (OrigVReg's class refined by
; the operand's subreg) and re-bases the destination subreg indices into that
; namespace. Regression = this compiles and passes -verify-machineinstrs.

define amdgpu_kernel void @uniform_add_i16(ptr addrspace(1) %result, ptr addrspace(1) %uniform.ptr, i16 %val) {
  %rmw = atomicrmw add ptr addrspace(1) %uniform.ptr, i16 %val monotonic, align 2
  store i16 %rmw, ptr addrspace(1) %result
  ret void
}

; CHECK-LABEL: uniform_add_i16:
; The previously-illegal descriptor slice is now a well-formed 64-bit-base load.
; CHECK: s_load_dword s{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0x0
; The i16 atomic is emitted as a cmpswap loop and the narrowed result stored back.
; CHECK: buffer_atomic_cmpswap
; CHECK: buffer_store_short
; CHECK: s_endpgm

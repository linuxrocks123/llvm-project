; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global \
; RUN:   -amdgpu-atomic-optimizer-strategy=Iterative -amdgpu-ssa-regalloc \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
;
; Regression test for two REG_SEQUENCE-lowering bugs in eliminateRegSequences
; (AMDGPUSSARegisterAllocator), both surfaced by a uniform i64 atomicrmw whose
; result is stored through a buffer-resource descriptor built by REG_SEQUENCE:
;
;   %d:sgpr_128    = REG_SEQUENCE %lo, %subreg.sub0, %hi, %subreg.sub1  ; 64-bit
;   %rsrc:sgpr_128 = REG_SEQUENCE %m1, %subreg.sub2,
;                                 %d,  %subreg.sub0_sub1,
;                                 %p,  %subreg.sub3
;
; Bug 1 (verifier crash): the over-wide source %d (sgpr_128 holding a 64-bit
;   value) was copied at full width into the 64-bit sub0_sub1 slice, emitting an
;   illegal width-mismatched copy ($sgpr0_sgpr1 = COPY $sgpr0_sgpr1_sgpr2_sgpr3).
;   Fixed by narrowing the source to its slice sub-register. Regression = this
;   compiles and passes -verify-machineinstrs at all.
;
; Bug 2 (miscompile): REG_SEQUENCE is a *parallel* assignment, but the slice
;   copies were emitted sequentially, so a slice overwriting a register still
;   needed by a later slice corrupted it (out.hi clobbered by out.lo via
;   "s_mov_b32 s1, s0"). Fixed by routing the copies through resolvePermutation.
;   Regression = the address high word (s1) is NOT overwritten with s0.

@local_var64 = external addrspace(3) global i64, align 8

define amdgpu_kernel void @add_i64_uniform(ptr addrspace(1) %out, i64 %additive) {
entry:
  %old = atomicrmw add ptr addrspace(3) @local_var64, i64 %additive acq_rel, align 8
  store i64 %old, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: add_i64_uniform:
; The out pointer is loaded into s[0:1] (low/high address words of the rsrc).
; CHECK: s_load_dwordx4 s[0:3], s[4:5]
; Bug 2 guard: the descriptor's high address word (s1 = out.hi) must be
; preserved — the parallel-copy lowering must NOT overwrite it with out.lo.
; CHECK-NOT: s_mov_b32 s1, s0
; The i64 result is stored through the assembled descriptor; no spilling.
; CHECK: buffer_store_dwordx2 v[0:1], off, s[0:3], 0
; CHECK: ScratchSize: 0

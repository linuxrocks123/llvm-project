; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s
;
; End-to-end SSA RA pipeline test (T1d): counted loop with an early-exit break.
;
; The 64-bit GEP index is sign-extended from a 32-bit loop-carried value, so
; RebuildSSA produces a PHI whose source names a SUBREGISTER:
;     %sext = PHI [%iv.sub0, entry], [%iv.sub0, latch]
; where the parent tuple is colored to an SGPR pair. PHI lowering must copy the
; corresponding 32-bit sub-physreg, not the full 64-bit tuple.
;
; Two bugs this guards:
;  1. lowerPHIs dropped the subreg index on PHI sources, emitting an illegal
;     64->32-bit copy (e.g. "s_mov_b32 s8, s[4:5]" / "; illegal copy s[4:5] to s8").
;  2. resolvePermutation took the register file from the first PHI of the block;
;     a block with mixed VGPR/SGPR PHIs and a same-file cycle could break the
;     cycle with wrong-file scratch/swaps.

define amdgpu_kernel void @pipeline_loop_exit(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) {
entry:
  br label %loop

loop:
  %i   = phi i32 [ 0, %entry ], [ %i.next, %cont ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %cont ]
  %p   = getelementptr i32, ptr addrspace(1) %in, i32 %i
  %v   = load i32, ptr addrspace(1) %p, align 4
  %brk = icmp eq i32 %v, -1
  br i1 %brk, label %exit, label %cont

cont:
  %acc.next = add i32 %acc, %v
  %i.next   = add i32 %i, 1
  %done = icmp uge i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  %r = phi i32 [ %acc, %loop ], [ %acc.next, %cont ]
  store i32 %r, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: pipeline_loop_exit:
; No malformed width-mismatched copy may be emitted anywhere.
; CHECK-NOT: illegal copy
;
; Loop header: sign-extend the index, scale by 4, load the element, test sentinel.
; CHECK: %loop
; CHECK: s_ashr_i32
; CHECK: s_lshl_b64
; CHECK: s_load_dword
; CHECK: s_cmp_eq_u32 s{{[0-9]+}}, -1
; CHECK: s_cbranch_scc0
;
; Exit stores the accumulated result.
; CHECK: %exit
; CHECK: global_store_dword

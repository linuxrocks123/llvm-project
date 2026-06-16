; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc < %s | FileCheck %s
;
; End-to-end SSA RA pipeline test (T1c): scalar counted loop.
;
; A loop-carried induction variable whose only use is its own update must NOT
; produce a copy on the back edge. RebuildSSA reconstructs:
;     %iv      = PHI [0, entry], [%iv.next, loop]
;     %iv.next = S_ADD %iv, step      ; %iv dies here (only use)
;     S_CMP %iv.next, n               ; reads the post-add value
; %iv dies at the S_ADD, so coloring reuses its physreg for %iv.next (identity
; PHI on the back edge). The loop must be a single block with the induction
; variable updated in place and no s_mov copy on the latch.
;
; This exercises the reachedByThisVNI reaching-definition fix: a position-only
; heuristic wrongly attributed the S_CMP use to the PHI value, which kept %iv
; live past the S_ADD and forced a spurious back-edge copy.

define amdgpu_kernel void @pipeline_loop(ptr addrspace(1) %out, i32 %n, i32 %step) {
entry:
  br label %loop

loop:
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %loop ]
  %i   = phi i32 [ 0, %entry ], [ %i.next,   %loop ]
  %acc.next = add i32 %acc, %step
  %i.next   = add i32 %i, %step
  %cmp = icmp ult i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  store i32 %acc.next, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: pipeline_loop:
; Induction variable initialized to 0 before the loop.
; CHECK: s_mov_b32 [[IV:s[0-9]+]], 0
; Single-block loop: in-place update (dst == src0), compare on the updated
; value, branch straight back. No s_mov copy on the back edge.
; CHECK: ; %loop
; CHECK: s_add_i32 [[IV]], [[IV]], s{{[0-9]+}}
; CHECK-NEXT: s_cmp_lt_u32 [[IV]], s{{[0-9]+}}
; CHECK-NEXT: s_cbranch_scc1 {{\.LBB[0-9]+_[0-9]+}}

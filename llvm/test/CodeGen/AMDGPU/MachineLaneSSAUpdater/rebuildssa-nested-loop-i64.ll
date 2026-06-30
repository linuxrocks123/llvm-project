; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
;
; Nested-loop companion to rebuildssa-loop-i64-reduction.ll. A loop-carried i64
; accumulator updated in the inner loop and live across the outer loop is
; lane-split (sub0/sub1) by RebuildSSA, whose IDF spans BOTH loop headers. The
; updater must create a header PHI in each and thread the inner PHI result into
; the outer header PHI's back-edge operand, while resolving each preheader
; operand to its reaching def (Part 1 rename-map fix). Regression = this compiles
; and passes -verify-machineinstrs (a non-Root lane preheader operand used to be
; left as an unresolved OrigVReg.subIdx placeholder -> no reaching def).

define amdgpu_kernel void @nested_i64(ptr addrspace(1) %out, i32 %n, i32 %m) {
entry:
  br label %outer

outer:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  %acc.o = phi i64 [ 0, %entry ], [ %acc.i.exit, %outer.latch ]
  br label %inner

inner:
  %j = phi i32 [ 0, %outer ], [ %j.next, %inner ]
  %acc.i = phi i64 [ %acc.o, %outer ], [ %acc.i.next, %inner ]
  %jext = zext i32 %j to i64
  %acc.i.next = add i64 %acc.i, %jext
  %j.next = add i32 %j, 1
  %jc = icmp slt i32 %j.next, %m
  br i1 %jc, label %inner, label %inner.exit

inner.exit:
  %acc.i.exit = phi i64 [ %acc.i.next, %inner ]
  %i.next = add i32 %i, 1
  br label %outer.latch

outer.latch:
  %ic = icmp slt i32 %i.next, %n
  br i1 %ic, label %outer, label %exit

exit:
  store i64 %acc.o, ptr addrspace(1) %out, align 8
  ret void
}

; CHECK-LABEL: nested_i64:
; Two nested loops must be present (outer depth 1, inner depth 2).
; CHECK: This Loop Header: Depth=1
; CHECK: This Inner Loop Header: Depth=2
; The loop-carried i64 accumulation (per-lane add/addc) survives SSA round-trip.
; CHECK: s_add_u32
; CHECK: s_addc_u32

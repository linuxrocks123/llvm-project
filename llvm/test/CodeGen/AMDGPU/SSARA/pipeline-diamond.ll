; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc < %s | FileCheck %s
;
; End-to-end SSA RA pipeline test (T1b): if/else diamond with a divergent PHI.
;
; Correctness properties verified:
;  1. Cross-width interference (WiderDefs): the wide argument load writes s[0:3];
;     the kernel-arg segment pointer must NOT be colored into that range. It
;     stays in its ABI pair s[4:5] and is reused by the out-ptr load. The buggy
;     coloring put the pointer in s[0:1] (subset of s[0:3]) and clobbered it.
;  2. Divergent control flow on the workitem id uses exec-mask save/restore.
;  3. The divergent PHI result is materialized into a VGPR for the store.

define amdgpu_kernel void @pipeline_diamond(ptr addrspace(1) %out, i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %cmp = icmp ne i32 %tid, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %sum_then = add i32 %a, %b
  %r_then = add i32 %sum_then, %c
  br label %merge

if.else:
  %sum_else = add i32 %c, %d
  %r_else = add i32 %sum_else, %b
  br label %merge

merge:
  %r = phi i32 [ %r_then, %if.then ], [ %r_else, %if.else ]
  %addr = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %r, ptr addrspace(1) %addr, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

; CHECK-LABEL: pipeline_diamond:
; Kernarg pointer stays in s[4:5], disjoint from the wide arg load's s[0:3] dest.
; CHECK: s_load_dwordx4 s[0:3], [[KARG:s\[4:5\]]], 0x2c
; Divergent branch on the workitem id.
; CHECK: s_and_saveexec_b64
; Divergent PHI result materialized into a VGPR.
; CHECK: v_mov_b32_e32 [[RES:v[0-9]+]], s{{[0-9]+}}
; Out-ptr load reuses the un-clobbered kernarg base.
; CHECK: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], [[KARG]], 0x24
; CHECK: global_store_dword v{{[0-9]+}}, [[RES]], s[{{[0-9]+:[0-9]+}}]

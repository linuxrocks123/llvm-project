; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc -stop-after=amdgpu-ssa-register-spiller -amdgpu-ssa-spill-markers=1 < %s | FileCheck --check-prefix=SPILLER %s
;
; End-to-end SSA RA pipeline test (T2c): loop with a spilled loop-invariant
; value. The precomputed scalar sum exceeds the VGPR budget (amdgpu-num-vgpr=5)
; and is spilled before the loop, reloaded on every iteration.

define amdgpu_kernel void @pipeline_spill_loop(ptr addrspace(1) %out, ptr addrspace(1) %in, i32 %n) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %base = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %b0 = load volatile i32, ptr addrspace(1) %base
  %q1 = getelementptr i32, ptr addrspace(1) %base, i32 1
  %b1 = load volatile i32, ptr addrspace(1) %q1
  %q2 = getelementptr i32, ptr addrspace(1) %base, i32 2
  %b2 = load volatile i32, ptr addrspace(1) %q2
  %q3 = getelementptr i32, ptr addrspace(1) %base, i32 3
  %b3 = load volatile i32, ptr addrspace(1) %q3
  br label %loop

loop:
  %i   = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc.next, %loop ]
  %p   = getelementptr i32, ptr addrspace(1) %base, i32 %i
  %v   = load volatile i32, ptr addrspace(1) %p
  %t0  = add i32 %v, %b0
  %t1  = add i32 %t0, %b1
  %t2  = add i32 %t1, %b2
  %t3  = add i32 %t2, %b3
  %acc.next = add i32 %acc, %t3
  %i.next = add i32 %i, 1
  %cmp = icmp ult i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  store i32 %acc.next, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
attributes #0 = { "amdgpu-num-vgpr"="5" }

; CHECK-LABEL: pipeline_spill_loop:
; CHECK: buffer_store_dword
; CHECK: buffer_load_dword
; CHECK: global_store_dword

; SPILLER-LABEL: name: pipeline_spill_loop
; SPILLER: SI_SPILL_V32_SAVE
; SPILLER: SI_VIRTUAL_SPILL_MARKER

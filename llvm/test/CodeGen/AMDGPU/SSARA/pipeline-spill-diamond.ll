; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc \
; RUN:   -stop-after=amdgpu-ssa-register-spiller -amdgpu-ssa-spill-markers=1 < %s \
; RUN:   | FileCheck %s --check-prefix=SPILLER
;
; End-to-end SSA RA pipeline test (T2b): if/else diamond with a PHI, forced to
; spill. Six inputs are live across the divergent region (two feed the arms, four
; are consumed after the merge); amdgpu-num-vgpr=6 forces a spill across the
; diamond, exercising spill/reload alongside exec-mask control flow and a PHI.

define amdgpu_kernel void @pipeline_spill_diamond(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p0 = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %p1 = getelementptr i32, ptr addrspace(1) %p0, i32 1
  %p2 = getelementptr i32, ptr addrspace(1) %p0, i32 2
  %p3 = getelementptr i32, ptr addrspace(1) %p0, i32 3
  %p4 = getelementptr i32, ptr addrspace(1) %p0, i32 4
  %p5 = getelementptr i32, ptr addrspace(1) %p0, i32 5
  %a0 = load volatile i32, ptr addrspace(1) %p0
  %a1 = load volatile i32, ptr addrspace(1) %p1
  %a2 = load volatile i32, ptr addrspace(1) %p2
  %a3 = load volatile i32, ptr addrspace(1) %p3
  %a4 = load volatile i32, ptr addrspace(1) %p4
  %a5 = load volatile i32, ptr addrspace(1) %p5
  %cmp = icmp ne i32 %tid, 0
  br i1 %cmp, label %then, label %else

then:
  %st = add i32 %a0, %a1
  br label %merge

else:
  %se = mul i32 %a0, %a1
  br label %merge

merge:
  %ph = phi i32 [ %st, %then ], [ %se, %else ]
  %r0 = add i32 %ph, %a2
  %r1 = add i32 %r0, %a3
  %r2 = add i32 %r1, %a4
  %r3 = add i32 %r2, %a5
  store i32 %r3, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
attributes #0 = { "amdgpu-num-vgpr"="6" }

; CHECK-LABEL: pipeline_spill_diamond:
; Tight budget forces a spill before the divergent region and a reload after it.
; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0{{.*}}Spill
; Divergent branch on the workitem id (diamond).
; CHECK: s_and_saveexec_b64
; CHECK: buffer_load_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0{{.*}}Reload
; CHECK: global_store_dword
; CHECK: NumVgprs: 6
; CHECK: ScratchSize: 8

; The spill is produced by the SSA spiller (virtual marker at the high-RP point).
; SPILLER-LABEL: name: pipeline_spill_diamond
; SPILLER: SI_SPILL_V32_SAVE
; SPILLER: SI_VIRTUAL_SPILL_MARKER

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc \
; RUN:   -stop-after=amdgpu-ssa-register-spiller -amdgpu-ssa-spill-markers=1 < %s \
; RUN:   | FileCheck %s --check-prefix=SPILLER
;
; End-to-end SSA RA pipeline test (T2a): straight-line kernel forced to spill.
;
; Six loaded values feed two independent reductions over ALL six inputs (an
; additive chain and a multiplicative chain), so every input stays live until
; both finish — minimum pressure is 5 VGPRs. With amdgpu-num-vgpr=4 the SSA
; spiller must spill: this exercises the full RebuildSSA -> Spiller -> SSA RA
; pipeline producing correct scratch spill/reload code under a tight budget.

define amdgpu_kernel void @pipeline_spill_linear(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
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
  %t0 = add i32 %a0, %a1
  %t1 = add i32 %t0, %a2
  %t2 = add i32 %t1, %a3
  %t3 = add i32 %t2, %a4
  %o0 = add i32 %t3, %a5
  %u0 = mul i32 %a0, %a1
  %u1 = mul i32 %u0, %a2
  %u2 = mul i32 %u1, %a3
  %u3 = mul i32 %u2, %a4
  %o1 = mul i32 %u3, %a5
  %r = add i32 %o0, %o1
  store i32 %r, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
attributes #0 = { "amdgpu-num-vgpr"="4" }

; CHECK-LABEL: pipeline_spill_linear:
; Tight VGPR budget forces spilling to scratch.
; CHECK: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0{{.*}}Spill
; CHECK: buffer_load_dword v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0{{.*}}Reload
; CHECK: global_store_dword
; Budget respected and scratch actually allocated.
; CHECK: NumVgprs: 4
; CHECK: ScratchSize: 12

; The spill is produced by the SSA spiller (virtual marker at the high-RP point).
; SPILLER-LABEL: name: pipeline_spill_linear
; SPILLER: SI_SPILL_V32_SAVE
; SPILLER: SI_VIRTUAL_SPILL_MARKER

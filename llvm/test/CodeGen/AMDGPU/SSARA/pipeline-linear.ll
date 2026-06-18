; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -amdgpu-ssa-regalloc < %s | FileCheck %s
;
; End-to-end SSA RA pipeline test (T1a): straight-line VGPR computation.
;
; Baseline no-CFG, no-spill case: a per-lane load feeds a short arithmetic
; chain that is stored back. The full pipeline must keep everything in VGPRs
; (divergent values from workitem id) and spill nothing.

define amdgpu_kernel void @pipeline_linear(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %addr = getelementptr i32, ptr addrspace(1) %in, i32 %tid
  %a = load i32, ptr addrspace(1) %addr, align 4
  %b = add i32 %a, %tid
  %c = mul i32 %b, %a
  %d = add i32 %c, %b
  %oaddr = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %d, ptr addrspace(1) %oaddr, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

; CHECK-LABEL: pipeline_linear:
; Loaded value feeds the VGPR arithmetic chain, result stored back.
; CHECK: global_load_dword [[A:v[0-9]+]], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}]
; CHECK: v_add_u32{{.*}}[[A]]
; CHECK: global_store_dword
; No spilling for this low-pressure linear kernel.
; CHECK: ScratchSize: 0

; RUN: llc -global-isel -mtriple=amdgcn -mcpu=gfx906 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for getLiveRegs skipping vregs without a live interval before querying
; their register kind (llvm::getLiveRegs in GCNRegPressure.cpp).
;
; A genuine spill in a GlobalISel function walks the live set via getLiveRegs,
; which called getRegKind (reads the register class) BEFORE the hasInterval
; check. Unused classless GlobalISel InstructionSelect leftovers have no class
; and no interval, so the class query asserted ("Register class not set"). The
; fix checks hasInterval first, skipping such vregs without touching their class.

; CHECK-LABEL: v256i8_liveout:
; CHECK: s_endpgm
define amdgpu_kernel void @v256i8_liveout(ptr addrspace(1) %src1, ptr addrspace(1) %src2, ptr addrspace(1) captures(none) %dst) {
entry:
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %gep1 = getelementptr <8 x i8>, ptr addrspace(1) %src1, i32 %idx
  %vec1 = load <256 x i8>, ptr addrspace(1) %gep1, align 256
  %gep2 = getelementptr <8 x i8>, ptr addrspace(1) %src2, i32 %idx
  %vec2 = load <256 x i8>, ptr addrspace(1) %gep2, align 256
  %cmp = icmp ult i32 %idx, 15
  br i1 %cmp, label %bb.1, label %bb.2
bb.1:
  br label %bb.2
bb.2:
  %tmp5 = phi <256 x i8> [ %vec1, %entry ], [ %vec2, %bb.1 ]
  store <256 x i8> %tmp5, ptr addrspace(1) %dst, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

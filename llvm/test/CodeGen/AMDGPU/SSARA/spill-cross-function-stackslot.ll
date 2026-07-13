; RUN: llc -amdgpu-scalarize-global-loads=false -mtriple=amdgcn -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for clearing the spiller's per-function stack-slot maps between functions
; (AMDGPUSSARegisterSpiller::runOnMachineFunction).
;
; Virt2StackSlotMap and StoredAtDefinition are keyed by VRegMaskPair, whose
; virtual register numbers restart in every function. They were never cleared,
; so a colliding {vreg, mask} in a later function returned a stale frame index
; valid only in the earlier function's MachineFrameInfo -> getObjectAlign
; "Invalid Object Idx". The two heavy-spilling kernels below reuse the same
; spilled {vreg, mask}; the second must not see the first's stale slot. Order
; matters: the first kernel populates the map, the second would crash.

; CHECK-LABEL: global_zextload_v64i16_to_v64i32:
; CHECK-LABEL: global_sextload_v64i16_to_v64i32:
; CHECK: s_endpgm
define amdgpu_kernel void @global_zextload_v64i16_to_v64i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %load = load <64 x i16>, ptr addrspace(1) %in, align 128
  %ext = zext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(1) %out, align 256
  ret void
}

define amdgpu_kernel void @global_sextload_v64i16_to_v64i32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %load = load <64 x i16>, ptr addrspace(1) %in, align 128
  %ext = sext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(1) %out, align 256
  ret void
}

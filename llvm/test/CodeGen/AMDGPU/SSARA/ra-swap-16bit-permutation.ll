; RUN: llc -mtriple=amdgcn -mcpu=gfx1100 -mattr=+real-true16 -amdgpu-ssa-regalloc -verify-machineinstrs < %s | FileCheck %s

; Guard for 16-bit permutation swaps in SSA destruction
; (AMDGPUSSARegisterAllocator::emitSwap).
;
; Two half loop-carried values swap each iteration, forming a permutation cycle
; that SSA destruction must break with a swap. On a true16 target the two values
; are packed into one VGPR's lo16/hi16, so the swap must use V_SWAP_B16 (16-bit
; operands). emitSwap routed any <=32-bit VGPR swap to V_SWAP_B32, whose operands
; are VGPR_32, producing an illegal 32-bit swap on 16-bit subregisters
; ("Operand has incorrect register class"). The fix emits V_SWAP_B16 on true16
; targets (with a 16-bit XOR triplet fallback).

; CHECK-LABEL: swap:
; CHECK: v_swap_b16
; CHECK: s_setpc_b64
define half @swap(half %a, half %b, i32 %i) {
entry:
  br label %loop

loop:
  %x = phi half [ %a, %entry ], [ %y, %loop ]
  %y = phi half [ %b, %entry ], [ %x, %loop ]
  %i2 = phi i32 [ %i, %entry ], [ %i3, %loop ]
  %i3 = sub i32 %i2, 1
  %cmp = icmp eq i32 %i3, 0
  br i1 %cmp, label %ret, label %loop

ret:
  ret half %x
}

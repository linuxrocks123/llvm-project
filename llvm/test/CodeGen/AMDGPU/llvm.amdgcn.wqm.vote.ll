; RUN: llc -global-isel=0 -mtriple=amdgpu8.02 < %s | FileCheck -check-prefixes=CHECK,WAVE64 %s
; xUN: llc -global-isel=1 -mtriple=amdgpu8.02 < %s | FileCheck -check-prefixes=CHECK,WAVE64 %s
; RUN: llc -global-isel=0 -mtriple=amdgpu10.10 < %s | FileCheck -check-prefixes=CHECK,WAVE32 %s
; xUN: llc -global-isel=1 -mtriple=amdgpu10.10 < %s | FileCheck -check-prefixes=CHECK,WAVE32 %s
; RUN: llc -global-isel=0 -mtriple=amdgpu11.00 < %s | FileCheck -check-prefixes=CHECK,WAVE32 %s
; xUN: llc -global-isel=1 -mtriple=amdgpu11.00 < %s | FileCheck -check-prefixes=CHECK,WAVE32 %s

define amdgpu_ps float @ret(i32 %v0, i32 %v1) #1 {
main_body:
  %c = icmp eq i32 %v0, %v1
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 %c)
  %r = select i1 %w, float 1.0, float 0.0
  ret float %r
}

define amdgpu_ps float @true() #1 {
main_body:
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 true)
  %r = select i1 %w, float 1.0, float 0.0
  ret float %r
}

define amdgpu_ps float @false() #1 {
main_body:
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 false)
  %r = select i1 %w, float 1.0, float 0.0
  ret float %r
}

; Note: an almost identical test for this exists in llvm.amdgcn.kill.ll
define amdgpu_ps float @kill(i32 %v0, i32 %v1) #1 {
main_body:
  %c = icmp eq i32 %v0, %v1
  %w = call i1 @llvm.amdgcn.wqm.vote(i1 %c)
  call void @llvm.amdgcn.kill(i1 %w)
  ret float 0.0
}

declare void @llvm.amdgcn.kill(i1) #1
declare i1 @llvm.amdgcn.wqm.vote(i1)

attributes #1 = { nounwind }

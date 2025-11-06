--- |
  ; ModuleID = '../../llvm/test/CodeGen/AMDGPU/SSASpiller/spill-use-before-spill.mir'
  source_filename = "../../llvm/test/CodeGen/AMDGPU/SSASpiller/spill-use-before-spill.mir"
  target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
  target triple = "amdgcn"
  
  define amdgpu_kernel void @test_use_before_spill(i32 %cond) #0 {
  entry:
    %cmp = icmp eq i32 %cond, 0
    br i1 %cmp, label %spill_path, label %clean_path
  
  spill_path:                                       ; preds = %entry
    br label %join
  
  clean_path:                                       ; preds = %entry
    br label %join
  
  join:                                             ; preds = %clean_path, %spill_path
    ret void
  }
  
  attributes #0 = { "amdgpu-num-vgpr"="8" "target-cpu"="gfx1200" }
...
---
name:            test_use_before_spill
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          false
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vreg_128, preferred-register: '', flags: [  ] }
  - { id: 1, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 2, class: vreg_64, preferred-register: '', flags: [  ] }
  - { id: 3, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 4, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 5, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 6, class: sreg_32, preferred-register: '', flags: [  ] }
  - { id: 7, class: vreg_128, preferred-register: '', flags: [  ] }
  - { id: 8, class: vreg_128, preferred-register: '', flags: [  ] }
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    4
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:
  - { id: 0, name: '', type: spill-slot, offset: 0, size: 16, alignment: 4, 
      stack-id: default, callee-saved-register: '', callee-saved-restored: true, 
      debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: true
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sgpr32'
  bytesInStackArgArea: 0
  returnsVoid:     true
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0.entry:
    successors: %bb.1(0x40000000), %bb.2(0x40000000)
    liveins: $vgpr0_vgpr1_vgpr2_vgpr3, $sgpr0
  
    %0:vreg_128 = COPY $vgpr0_vgpr1_vgpr2_vgpr3
    %3:sreg_32 = COPY $sgpr0
    S_CMP_EQ_U32 %3, 0, implicit-def $scc
    S_CBRANCH_SCC1 %bb.1, implicit $scc
    S_BRANCH %bb.2
  
  bb.1.spill_path:
    successors: %bb.3(0x80000000)
  
    S_NOP 0, implicit %0
    %1:vreg_64 = IMPLICIT_DEF
    SI_SPILL_V128_SAVE %0, %stack.0, $sgpr32, 0, implicit $exec :: (store (s128) into %stack.0, align 4, addrspace 5)
    %2:vreg_64 = IMPLICIT_DEF
    S_NOP 0, implicit %1, implicit %2
    %4:sreg_32 = S_MOV_B32 1
    S_BRANCH %bb.3
  
  bb.2.clean_path:
    successors: %bb.3(0x80000000)
  
    %5:sreg_32 = S_MOV_B32 0
    S_BRANCH %bb.3
  
  bb.3.join:
    successors: %bb.5(0x40000000), %bb.4(0x40000000)
  
    %6:sreg_32 = PHI %4, %bb.1, %5, %bb.2
    S_CMP_EQ_U32 %6, 0, implicit-def $scc
    S_CBRANCH_SCC1 %bb.5, implicit $scc
  
  bb.4.join:
    successors: %bb.5(0x80000000)
  
    %7:vreg_128 = SI_SPILL_V128_RESTORE %stack.0, $sgpr32, 0, implicit $exec :: (load (s128) from %stack.0, align 4, addrspace 5)
  
  bb.5.join:
    %8:vreg_128 = PHI %7, %bb.4, %0, %bb.3
    S_ENDPGM 0, implicit %8
...

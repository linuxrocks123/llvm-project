//===-- AMDGPUSSARegisterAllocator.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// SSA-based Register Allocator for AMDGPU.
///
/// Implements width-descending multi-pass PEO coloring based on:
/// "Register Allocation for Programs in SSA-Form"
/// Sebastian Hack, Daniel Grund, Gerhard Goos (CC'06)
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERALLOCATOR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERALLOCATOR_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include <set>

namespace llvm {

class GCNSubtarget;
class SIRegisterInfo;
class SIInstrInfo;

class AMDGPUSSARegisterAllocator : public MachineFunctionPass {
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineDominatorTree *MDT = nullptr;
  LiveIntervals *LIS = nullptr;
  MachineLoopInfo *MLI = nullptr;
  SlotIndexes *Indexes = nullptr;
  const GCNSubtarget *ST = nullptr;
  RegisterClassInfo RegClassInfo;

  std::set<unsigned, std::greater<unsigned>> ColoringOrder;
  DenseMap<Register, MCRegister> ColorMap;
  BitVector OccupiedRegUnits;

  void classifyVRegs();
  void colorAndSplit(MachineFunction &MF);
  void colorByWidth(unsigned Width);
  void seedOccupiedAtBBEntry(MachineBasicBlock *MBB);
  void markOccupied(MCRegister PhysReg);
  void markFree(MCRegister PhysReg);
  MCRegister pickFreePhysReg(const TargetRegisterClass *RC);

  // === Split+Assign for occupancy improvement ===

  struct ShadowEntry {
    Register VReg;       // real vreg, or invalid for split fragments
    MCRegister PhysReg;
    SlotIndex Start;
    SlotIndex End;
    unsigned Width;      // in 32-bit units
  };
  using ShadowMap = SmallVector<ShadowEntry>;

  struct SplitPlan {
    Register VReg;
    SlotIndex SplitPoint;
    MCRegister NewPhysReg;
  };

  struct RecolorPlan {
    Register VReg;
    MCRegister OldPhysReg;
    MCRegister NewPhysReg;
  };

  struct Gap {
    unsigned HWIndex;
    unsigned Width;
    SlotIndex Start;
    SlotIndex End;
  };

  struct OccupancyPlan {
    SmallVector<SplitPlan> Splits;
    SmallVector<RecolorPlan> Recolors;
    unsigned NewMaxIndex = 0;
  };

  bool splitForOccupancy(MachineFunction &MF);
  bool planSplit(OccupancyPlan &Plan, unsigned TargetMaxIndex);
  void commitPlan(MachineFunction &MF, const OccupancyPlan &Plan);
  unsigned computeMaxPhysRegIndex(bool IsVGPR) const;
  bool isPhysRegFreeForRange(MCRegister PhysReg, SlotIndex Start, SlotIndex End,
                             const ShadowMap &Shadow) const;
  MCRegister findFreeInRange(const TargetRegisterClass *RC,
                             SlotIndex Start, SlotIndex End,
                             unsigned FreedStart, unsigned FreedEnd,
                             const ShadowMap &Shadow) const;
  void fillGaps(SmallVectorImpl<Gap> &Gaps, unsigned TargetMaxIndex,
                ShadowMap &Shadow, OccupancyPlan &Plan);

public:
  static char ID;

  AMDGPUSSARegisterAllocator() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU SSA Register Allocator";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERALLOCATOR_H

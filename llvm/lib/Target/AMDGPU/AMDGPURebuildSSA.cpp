//===-- AMDGPURebuildSSA.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reconstruct SSA form from non-SSA MIR by splitting multi-def vregs into
// single-def vregs and inserting PHI nodes.
//
// This is a temporary bridge: converts post-PHIElimination MIR back to SSA
// so the SSA Register Allocator can run.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-rebuild-ssa"

namespace {

class AMDGPURebuildSSALegacy : public MachineFunctionPass {
  LiveIntervals *LIS = nullptr;
  MachineDominatorTree *MDT = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineLoopInfo *MLI = nullptr;

  VNInfo *incomingOnEdge(LiveInterval &LI, MachineInstr *Phi,
                         MachineOperand &PhiOp) {
    unsigned OpIdx = Phi->getOperandNo(&PhiOp);
    MachineBasicBlock *Pred = Phi->getOperand(OpIdx + 1).getMBB();
    SlotIndex EndB = LIS->getMBBEndIdx(Pred);
    return LI.getVNInfoBefore(EndB);
  }

  bool reachedByThisVNI(LiveInterval &LI, MachineInstr *DefMI,
                        MachineInstr *UseMI, MachineOperand &UseOp,
                        VNInfo *VNI) {
    if (UseMI->isPHI())
      return incomingOnEdge(LI, UseMI, UseOp) == VNI;

    if (UseMI->getParent() == DefMI->getParent()) {
      SlotIndex DefIdx = LIS->getInstructionIndex(*DefMI);
      SlotIndex UseIdx = LIS->getInstructionIndex(*UseMI);
      return DefIdx < UseIdx;
    }
    return MDT->dominates(DefMI->getParent(), UseMI->getParent());
  }

  LaneBitmask operandLaneMask(const MachineOperand &MO) const {
    if (unsigned Sub = MO.getSubReg())
      return TRI->getSubRegIndexLaneMask(Sub);
    return MRI->getMaxLaneMaskForVReg(MO.getReg());
  }

  Register buildRSForSuperUse(MachineInstr *UseMI, MachineOperand &MO,
                              Register OldVR, Register NewVR,
                              LaneBitmask MaskToRewrite, LiveInterval &LI,
                              const TargetRegisterClass *OpRC,
                              SlotIndex &OutIdx,
                              SmallVectorImpl<LaneBitmask> &LanesToExtend) {
    MachineBasicBlock *InsertBB = UseMI->getParent();
    MachineBasicBlock::iterator IP(UseMI);
    SlotIndex QueryIdx;

    if (UseMI->isPHI()) {
      unsigned OpIdx = UseMI->getOperandNo(&MO);
      MachineBasicBlock *Pred = UseMI->getOperand(OpIdx + 1).getMBB();
      InsertBB = Pred;
      IP = Pred->getFirstTerminator();
      QueryIdx = LIS->getMBBEndIdx(Pred).getPrevSlot();
    } else {
      QueryIdx = LIS->getInstructionIndex(*UseMI);
    }

    Register Dest = MRI->createVirtualRegister(OpRC);
    auto RS = BuildMI(*InsertBB, IP,
                      (IP != InsertBB->end() ? IP->getDebugLoc() : DebugLoc()),
                      TII->get(TargetOpcode::REG_SEQUENCE), Dest);

    SmallDenseSet<unsigned, 8> AddedSubIdxs;
    SmallDenseSet<LaneBitmask::Type, 8> AddedMasks;

    for (const LiveInterval::SubRange &SR : LI.subranges()) {
      if (!SR.getVNInfoAt(QueryIdx))
        continue;
      LaneBitmask Lane = SR.LaneMask;
      if (!AddedMasks.insert(Lane.getAsInteger()).second)
        continue;

      unsigned SubIdx = TRI->getSubRegIndexForLaneMask(Lane);
      if (!SubIdx || !AddedSubIdxs.insert(SubIdx).second)
        continue;

      if (Lane == MaskToRewrite)
        RS.addReg(NewVR).addImm(SubIdx);
      else
        RS.addReg(OldVR, 0, SubIdx).addImm(SubIdx);

      LanesToExtend.push_back(Lane);
    }

    if (AddedSubIdxs.empty()) {
      unsigned SubIdx = TRI->getSubRegIndexForLaneMask(MaskToRewrite);
      RS.addReg(NewVR).addImm(SubIdx);
      LanesToExtend.push_back(MaskToRewrite);
    }

    LIS->InsertMachineInstrInMaps(*RS);
    OutIdx = LIS->getInstructionIndex(*RS);

    LLVM_DEBUG({
      dbgs() << "  [RS] inserted ";
      RS->print(dbgs());
    });
    return Dest;
  }

  void extendAt(LiveInterval &LI, SlotIndex Idx, ArrayRef<LaneBitmask> Lanes) {
    SmallVector<SlotIndex, 1> P{Idx};
    LIS->extendToIndices(LI, P);
    for (auto &SR : LI.subranges())
      for (LaneBitmask L : Lanes)
        if (SR.LaneMask == L)
          LIS->extendToIndices(SR, P);
  }

  void buildRealPHI(VNInfo *VNI, LiveInterval &LI, Register OldVR);
  void splitNonPhiValue(VNInfo *VNI, LiveInterval &LI, Register OldVR);
  void rewriteUses(MachineInstr *DefMI, Register OldVR,
                   LaneBitmask MaskToRewrite, Register NewVR, LiveInterval &LI,
                   VNInfo *VNI);

public:
  static char ID;
  AMDGPURebuildSSALegacy() : MachineFunctionPass(ID) {
    initializeAMDGPURebuildSSALegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitiveID(MachineDominatorsID);
    AU.addPreservedID(MachineDominatorsID);
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

// === buildRealPHI ===

void AMDGPURebuildSSALegacy::buildRealPHI(VNInfo *VNI, LiveInterval &LI,
                                          Register OldVR) {
  MachineBasicBlock *DefMBB = LIS->getMBBFromIndex(VNI->def);
  SmallVector<MachineOperand> Ops;
  const LaneBitmask FullMask = MRI->getMaxLaneMaskForVReg(OldVR);

  LaneBitmask CommonMask = LaneBitmask::getAll();
  LaneBitmask UnionMask = LaneBitmask::getNone();

  LLVM_DEBUG(dbgs() << "\n[PHI] Build PHI for " << printReg(OldVR)
                    << " at MBB_" << DefMBB->getNumber() << '\n');

  for (auto *Pred : DefMBB->predecessors()) {
    SlotIndex EndB = LIS->getMBBEndIdx(Pred);
    LaneBitmask EdgeMask = LaneBitmask::getNone();

    for (const LiveInterval::SubRange &SR : LI.subranges())
      if (SR.getVNInfoBefore(EndB))
        EdgeMask |= SR.LaneMask;

    LLVM_DEBUG({
      const bool HasSubranges = !LI.subranges().empty();
      VNInfo *MainOut = LI.getVNInfoBefore(EndB);
      dbgs() << "    subranges: " << (HasSubranges ? "yes" : "no")
             << ", main-range live-out: " << (MainOut ? "yes" : "no") << '\n';
    });

    if (EdgeMask.none()) {
      LLVM_DEBUG({
        dbgs() << "    EdgeMask is NONE; reason: ";
        if (LI.subranges().empty())
          dbgs() << "no subranges for this vreg";
        else if (LI.getVNInfoBefore(EndB))
          dbgs() << "subranges exist but none live at edge; main-range is "
                    "live-out";
        else
          dbgs() << "subranges exist and main-range not live-out (treating as "
                    "undef edge)";
        dbgs() << "\n";
      });

      if (LI.subranges().empty() || LI.getVNInfoBefore(EndB))
        EdgeMask = FullMask;
    }
    CommonMask &= EdgeMask;
    UnionMask |= EdgeMask;

    unsigned SubIdx = AMDGPU::NoRegister;
    if ((FullMask & ~EdgeMask).any())
      SubIdx = TRI->getSubRegIndexForLaneMask(EdgeMask);

    Ops.push_back(MachineOperand::CreateReg(OldVR, /*isDef*/ false,
                                            /*isImp*/ false, /*isKill*/ false,
                                            /*isDead*/ false, /*isUndef*/ false,
                                            /*isEarlyClobber*/ false, SubIdx));
    Ops.push_back(MachineOperand::CreateMBB(Pred));
  }

  LaneBitmask PhiMask = (CommonMask.none() ? UnionMask : CommonMask);
  if (PhiMask.none())
    PhiMask = FullMask;

  LLVM_DEBUG(dbgs() << "  [PHI] final mask=" << PrintLaneMask(PhiMask) << '\n');

  const TargetRegisterClass *RC =
      TRI->getRegClassForOperandReg(*MRI, Ops.front());
  Register DestReg = MRI->createVirtualRegister(RC);

  auto PHINode = BuildMI(*DefMBB, DefMBB->begin(), DebugLoc(),
                         TII->get(TargetOpcode::PHI), DestReg)
                     .add(ArrayRef(Ops));
  MachineInstr *PHI = PHINode.getInstr();
  LIS->InsertMachineInstrInMaps(*PHI);

  LLVM_DEBUG({
    dbgs() << "  [PHI] inserted ";
    PHI->print(dbgs());
  });

  rewriteUses(PHI, OldVR, PhiMask, DestReg, LI, VNI);
  LIS->createAndComputeVirtRegInterval(DestReg);
}

// === splitNonPhiValue ===

void AMDGPURebuildSSALegacy::splitNonPhiValue(VNInfo *VNI, LiveInterval &LI,
                                              Register OldVR) {
  MachineInstr *DefMI = LIS->getInstructionFromIndex(VNI->def);
  int OpIdx = DefMI->findRegisterDefOperandIdx(OldVR, TRI, /*IsDead*/ false,
                                               /*Overlaps*/ true);
  MachineOperand &MO = DefMI->getOperand(OpIdx);
  unsigned SubRegIdx = MO.getSubReg();

  LaneBitmask Mask = SubRegIdx ? TRI->getSubRegIndexLaneMask(SubRegIdx)
                               : MRI->getMaxLaneMaskForVReg(MO.getReg());
  const TargetRegisterClass *RC = TRI->getRegClassForOperandReg(*MRI, MO);

  Register NewVR = MRI->createVirtualRegister(RC);
  MO.setReg(NewVR);
  MO.setSubReg(AMDGPU::NoRegister);
  MO.setIsUndef(false);
  LIS->ReplaceMachineInstrInMaps(*DefMI, *DefMI);

  LLVM_DEBUG({
    dbgs() << "[SPLIT] def ";
    DefMI->print(dbgs());
    dbgs() << "        lanes=" << PrintLaneMask(Mask) << " -> new vreg "
           << printReg(NewVR) << '\n';
  });

  rewriteUses(DefMI, OldVR, Mask, NewVR, LI, VNI);
  LIS->createAndComputeVirtRegInterval(NewVR);
}

// === rewriteUses ===

void AMDGPURebuildSSALegacy::rewriteUses(MachineInstr *DefMI, Register OldVR,
                                         LaneBitmask MaskToRewrite,
                                         Register NewVR, LiveInterval &LI,
                                         VNInfo *VNI) {
  const TargetRegisterClass *NewRC = TRI->getRegClassForReg(*MRI, NewVR);

  LLVM_DEBUG(dbgs() << "[RW] rewriting uses of " << printReg(OldVR)
                    << " lanes=" << PrintLaneMask(MaskToRewrite) << " with "
                    << printReg(NewVR) << '\n');

  for (MachineOperand &MO :
       llvm::make_early_inc_range(MRI->use_operands(OldVR))) {
    MachineInstr *UseMI = MO.getParent();
    if (UseMI == DefMI)
      continue;

    if (!reachedByThisVNI(LI, DefMI, UseMI, MO, VNI))
      continue;

    LaneBitmask OpMask = operandLaneMask(MO);
    if ((OpMask & MaskToRewrite).none())
      continue;

    const TargetRegisterClass *OpRC = TRI->getRegClassForOperandReg(*MRI, MO);

    if (OpMask == MaskToRewrite &&
        isOfRegClass(getRegSubRegPair(MO), *NewRC, *MRI)) {
      LLVM_DEBUG(dbgs() << "  [RW] exact -> " << printReg(NewVR) << " at ";
                 UseMI->print(dbgs()));
      MO.setReg(NewVR);
      MO.setSubReg(AMDGPU::NoRegister);
      continue;
    }

    if ((OpMask & ~MaskToRewrite).any()) {
      SmallVector<LaneBitmask, 4> LanesToExtend;
      SlotIndex RSIdx;
      Register RSv = buildRSForSuperUse(UseMI, MO, OldVR, NewVR, MaskToRewrite,
                                        LI, OpRC, RSIdx, LanesToExtend);
      extendAt(LI, RSIdx, LanesToExtend);
      MO.setReg(RSv);
      MO.setSubReg(AMDGPU::NoRegister);
    } else {
      unsigned Sub = MO.getSubReg();
      assert(Sub && "subset path requires a subregister use");
      LLVM_DEBUG(dbgs() << "  [RW] subset sub" << Sub << " -> "
                        << printReg(NewVR) << " at ";
                 UseMI->print(dbgs()));
      MO.setReg(NewVR);
      MO.setSubReg(Sub);
    }
  }
}

// === runOnMachineFunction ===

bool AMDGPURebuildSSALegacy::runOnMachineFunction(MachineFunction &MF) {
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  if (MRI->isSSA())
    return false;

  LLVM_DEBUG(dbgs() << "\n=== AMDGPURebuildSSA on " << MF.getName()
                    << " ===\n");

  DenseMap<MachineBasicBlock *, unsigned> DomPreorder;
  {
    unsigned N = 0;
    for (auto *Node : depth_first(MDT->getRootNode()))
      DomPreorder[Node->getBlock()] = N++;
  }

  DenseSet<Register> Processed;

  for (auto &B : MF) {
    for (auto &I : B) {
      for (auto Def : I.defs()) {
        if (!Def.isReg() || !Def.getReg().isVirtual())
          continue;

        Register VReg = Def.getReg();
        if (!LIS->hasInterval(VReg) || !Processed.insert(VReg).second)
          continue;

        LiveInterval &LI = LIS->getInterval(VReg);
        if (LI.getNumValNums() == 1)
          continue;

        LLVM_DEBUG(dbgs() << "\n[VREG] " << printReg(VReg) << " has "
                          << LI.getNumValNums() << " VNs\n");

        SmallVector<VNInfo *, 8> WorkList;
        for (VNInfo *V : LI.vnis())
          if (V && !V->isUnused())
            WorkList.push_back(V);

        llvm::sort(WorkList, [&](VNInfo *A, VNInfo *B) {
          MachineBasicBlock *BBA = LIS->getMBBFromIndex(A->def);
          MachineBasicBlock *BBB = LIS->getMBBFromIndex(B->def);
          if (DomPreorder[BBA] != DomPreorder[BBB])
            return DomPreorder[BBA] < DomPreorder[BBB];
          return A->def < B->def;
        });

        LLVM_DEBUG({
          dbgs() << "  [WL] order:\n";
          for (VNInfo *V : WorkList)
            dbgs() << "    id=" << V->id << " def=" << V->def
                   << (V->isPHIDef() ? " (phi)\n" : "\n");
        });

        VNInfo *Root = WorkList.front();
        auto IsPhi = [&](VNInfo *V) { return V != Root && V->isPHIDef(); };
        auto Mid =
            std::stable_partition(WorkList.begin(), WorkList.end(), IsPhi);

        auto PHISlice =
            llvm::ArrayRef(WorkList).take_front(Mid - WorkList.begin());
        for (auto It = PHISlice.rbegin(); It != PHISlice.rend(); ++It)
          buildRealPHI(*It, LI, VReg);

        for (VNInfo *VNI :
             llvm::ArrayRef(WorkList).slice(Mid - WorkList.begin())) {
          if (VNI == Root)
            continue;
          splitNonPhiValue(VNI, LI, VReg);
        }

        // FIXME: shrinkToUses makes REG_SEQUENCE use definitions dead.
        LI.RenumberValues();
      }
    }
  }

  Processed.clear();

  MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
  MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);

  LLVM_DEBUG({
    dbgs() << "=== verify ===\n";
    MF.verify();
  });
  return MRI->isSSA();
}

char AMDGPURebuildSSALegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                    false, false)

FunctionPass *llvm::createAMDGPURebuildSSALegacyPass() {
  return new AMDGPURebuildSSALegacy();
}

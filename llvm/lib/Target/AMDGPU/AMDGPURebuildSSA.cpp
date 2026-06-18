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
#include "llvm/CodeGen/MachineLaneSSAUpdater.h"
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

public:
  static char ID;
  AMDGPURebuildSSALegacy() : MachineFunctionPass(ID) {
    initializeAMDGPURebuildSSALegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequiredTransitiveID(MachineDominatorsID);
    AU.addPreservedID(MachineDominatorsID);
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

// === runOnMachineFunction ===

bool AMDGPURebuildSSALegacy::runOnMachineFunction(MachineFunction &MF) {
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();

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

  MachineLaneSSAUpdater Updater(MF, *LIS, *MDT, *TRI);
  DenseSet<Register> Processed;
  SmallVector<MachineOperand *, 4> PHIDefs;

  for (auto &B : MF) {
    for (auto &I : B) {
      for (auto Def : I.defs()) {
        if (!Def.isReg() || !Def.getReg().isVirtual())
          continue;

        Register VReg = Def.getReg();
        if (!LIS->hasInterval(VReg) || !Processed.insert(VReg).second)
          continue;

        // Collect VNInfos before any repair calls: repairSSAForNewDef
        // replaces OrigVReg's interval object, invalidating any reference
        // held across calls.
        LiveInterval &LI = LIS->getInterval(VReg);
        if (LI.getNumValNums() == 1)
          continue;

        LLVM_DEBUG(dbgs() << "\n[VREG] " << printReg(VReg) << " has "
                          << LI.getNumValNums() << " VNs\n");

        // Find the earliest non-PHI definition in dom-preorder.
        // This is the unique original SSA def that anchors the vreg.
        VNInfo *Root = nullptr;
        for (VNInfo *V : LI.vnis()) {
          if (!V || V->isUnused() || V->isPHIDef())
            continue;
          if (!Root ||
              DomPreorder[LIS->getMBBFromIndex(V->def)] <
              DomPreorder[LIS->getMBBFromIndex(Root->def)])
            Root = V;
        }

        // Collect re-defs (non-Root, non-PHI) in dom-preorder, Root last.
        // PHI VNInfos carry no instruction; they are handled implicitly
        // by the IDF computation inside repairSSAForNewDef.
        // Root is placed last so the updater sees OrigVReg's interval
        // reduced to Root's lanes only when computing Root's IDF.
        SmallVector<VNInfo *, 8> WorkList;
        for (VNInfo *V : LI.vnis())
          if (V && !V->isUnused() && !V->isPHIDef() && V != Root)
            WorkList.push_back(V);
        llvm::sort(WorkList, [&](VNInfo *A, VNInfo *B) {
          MachineBasicBlock *BBA = LIS->getMBBFromIndex(A->def);
          MachineBasicBlock *BBB = LIS->getMBBFromIndex(B->def);
          if (DomPreorder[BBA] != DomPreorder[BBB])
            return DomPreorder[BBA] < DomPreorder[BBB];
          return A->def < B->def;
        });
        WorkList.push_back(Root);
        // LI not used below this point.

        assert(Root && "live interval with multiple VNs must have a non-PHI def");

        // Rename each re-def to a fresh vreg. repairSSAForNewDef inserts
        // lane-aware PHIs at pruned IDF blocks and rewrites dominated uses
        // with exact/subset/super policy, then recomputes LiveIntervals for
        // all affected vregs.
        // A full-register Root def is the unique SSA def for its lanes and
        // needs no renaming. A partial (subreg) Root def leaves OrigVReg as
        // a wide register class with only a subset of lanes live, causing
        // downstream passes to allocate the full register tuple unnecessarily.
        for (VNInfo *VNI : WorkList) {
          MachineInstr *DefMI = LIS->getInstructionFromIndex(VNI->def);
          assert(DefMI && "non-PHI VNInfo must have an instruction");
          if (VNI == Root) {
            int Idx = DefMI->findRegisterDefOperandIdx(VReg, TRI,
                                                       /*IsDead=*/false,
                                                       /*Overlaps=*/true);
            if (Idx < 0 || !DefMI->getOperand(Idx).getSubReg())
              continue;
          }
          PHIDefs.clear();
          Updater.repairSSAForNewDef(*DefMI, VReg, PHIDefs);
        }
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
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPURebuildSSALegacy, DEBUG_TYPE, "AMDGPU Rebuild SSA",
                    false, false)

FunctionPass *llvm::createAMDGPURebuildSSALegacyPass() {
  return new AMDGPURebuildSSALegacy();
}

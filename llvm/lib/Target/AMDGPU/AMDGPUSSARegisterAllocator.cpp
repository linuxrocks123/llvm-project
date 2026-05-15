//===-- AMDGPUSSARegisterAllocator.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSSARegisterAllocator.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineLaneSSAUpdater.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-allocator"

char AMDGPUSSARegisterAllocator::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUSSARegisterAllocator, DEBUG_TYPE,
                      "AMDGPU SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUSSARegisterAllocator, DEBUG_TYPE,
                    "AMDGPU SSA Register Allocator", false, false)

void AMDGPUSSARegisterAllocator::classifyVRegs() {
  ColoringOrder.clear();
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I < E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(VReg))
      continue;
    ColoringOrder.insert(TRI->getRegSizeInBits(*MRI->getRegClass(VReg)));
  }

  LLVM_DEBUG({
    dbgs() << "Coloring order (width descending):";
    for (unsigned W : ColoringOrder)
      dbgs() << " " << W;
    dbgs() << "\n";
  });
}

void AMDGPUSSARegisterAllocator::markOccupied(MCRegister PhysReg) {
  for (MCRegUnit Unit : TRI->regunits(PhysReg))
    OccupiedRegUnits.set(Unit);
}

void AMDGPUSSARegisterAllocator::markFree(MCRegister PhysReg) {
  for (MCRegUnit Unit : TRI->regunits(PhysReg))
    OccupiedRegUnits.reset(Unit);
}

MCRegister AMDGPUSSARegisterAllocator::pickFreePhysReg(
    const TargetRegisterClass *RC) {
  LLVM_DEBUG({
    dbgs() << "    Allocation order for " << TRI->getRegClassName(RC) << ":";
    for (MCRegister PR : RegClassInfo.getOrder(RC))
      dbgs() << " " << TRI->getName(PR);
    dbgs() << "\n";
  });

  for (MCRegister PR : RegClassInfo.getOrder(RC)) {
    bool Free = true;
    for (MCRegUnit Unit : TRI->regunits(PR))
      if (OccupiedRegUnits.test(Unit)) { Free = false; break; }
    if (Free)
      return PR;
  }
  return MCRegister();
}

void AMDGPUSSARegisterAllocator::seedOccupiedAtBBEntry(MachineBasicBlock *MBB) {
  OccupiedRegUnits.reset();
  SlotIndex BBStart = LIS->getMBBStartIdx(MBB);

  LLVM_DEBUG(dbgs() << "  Seed " << printMBBReference(*MBB) << ":\n");

  for (const auto &[VReg, PhysReg] : ColorMap) {
    if (LIS->getInterval(VReg).liveAt(BBStart)) {
      markOccupied(PhysReg);
      LLVM_DEBUG(dbgs() << "    live-in: " << printReg(VReg, TRI)
                        << " -> " << TRI->getName(PhysReg) << "\n");
    }
  }

  for (const auto &LI : MBB->liveins()) {
    markOccupied(LI.PhysReg);
    LLVM_DEBUG(dbgs() << "    phys live-in: " << TRI->getName(LI.PhysReg)
                      << "\n");
  }
}

void AMDGPUSSARegisterAllocator::colorByWidth(unsigned Width) {
  LLVM_DEBUG(dbgs() << "\n=== Width pass: " << Width << "-bit ===\n");

  for (auto *Node : depth_first(MDT->getRootNode())) {
    MachineBasicBlock *MBB = Node->getBlock();
    seedOccupiedAtBBEntry(MBB);

    for (MachineInstr &MI : *MBB) {
      for (MachineOperand &MO : MI.defs()) {
        Register VReg = MO.getReg();
        if (!VReg.isVirtual())
          continue;

        // Width != current pass: if wider, it was colored in a previous
        // pass and must be marked occupied at its def point (not live-in,
        // born in this BB). If narrower, it hasn't been colored yet —
        // ColorMap.find() returns end(), nothing to mark.
        if (TRI->getRegSizeInBits(*MRI->getRegClass(VReg)) != Width) {
          if (auto It = ColorMap.find(VReg); It != ColorMap.end()) {
            markOccupied(It->second);
            LLVM_DEBUG(dbgs() << "    mark wider def: "
                              << printReg(VReg, TRI) << " -> "
                              << TRI->getName(It->second) << "\n");
          }
          continue;
        }

        MCRegister Chosen;
        unsigned UseOpIdx;
        if (MI.isRegTiedToUseOperand(MO.getOperandNo(), &UseOpIdx)) {
          Chosen = ColorMap.lookup(MI.getOperand(UseOpIdx).getReg());
          assert(Chosen && "Tied use must be colored already");
          LLVM_DEBUG(dbgs() << "    tied: " << printReg(VReg, TRI)
                            << " inherits " << TRI->getName(Chosen) << "\n");
        } else {
          Chosen = pickFreePhysReg(MRI->getRegClass(VReg));
          assert(Chosen && "Failed to find free physreg");
          LLVM_DEBUG(dbgs() << "    color: " << printReg(VReg, TRI)
                            << " -> " << TRI->getName(Chosen) << "\n");
        }

        ColorMap[VReg] = Chosen;
        markOccupied(Chosen);
      }

      SlotIndex NextSI = LIS->getInstructionIndex(MI).getRegSlot().getNextSlot();
      for (const MachineOperand &MO : MI.uses()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        auto It = ColorMap.find(MO.getReg());
        if (It == ColorMap.end())
          continue;
        if (!LIS->getInterval(MO.getReg()).liveAt(NextSI)) {
          markFree(It->second);
          LLVM_DEBUG(dbgs() << "    kill: " << printReg(MO.getReg(), TRI)
                            << " free " << TRI->getName(It->second) << "\n");
        }
      }
    }
  }
}

void AMDGPUSSARegisterAllocator::colorAndSplit(MachineFunction &MF) {
  for (unsigned Width : ColoringOrder) {
    colorByWidth(Width);
    splitForOccupancy(MF);
  }

  LLVM_DEBUG({
    dbgs() << "\nColoring result:\n";
    for (const auto &[VReg, PhysReg] : ColorMap)
      dbgs() << "  " << printReg(VReg, TRI) << " -> "
             << TRI->getName(PhysReg) << "\n";
  });
}

// === Split+Assign for occupancy improvement ===

unsigned AMDGPUSSARegisterAllocator::computeMaxPhysRegIndex(bool IsVGPR) const {
  unsigned MaxIdx = 0;
  for (const auto &[VReg, PhysReg] : ColorMap) {
    const TargetRegisterClass *RC = MRI->getRegClass(VReg);
    if (TRI->isVGPRClass(RC) != IsVGPR)
      continue;
    unsigned Idx = TRI->getHWRegIndex(PhysReg);
    unsigned Width = TRI->getRegSizeInBits(*RC) / 32;
    MaxIdx = std::max(MaxIdx, Idx + Width);
  }
  return MaxIdx;
}

bool AMDGPUSSARegisterAllocator::isPhysRegFreeForRange(
    MCRegister PhysReg, SlotIndex Start, SlotIndex End,
    const ShadowMap &Shadow) const {
  for (const auto &E : Shadow) {
    if (!TRI->regsOverlap(PhysReg, E.PhysReg))
      continue;
    if (Start < E.End && End > E.Start)
      return false;
  }
  return true;
}

MCRegister AMDGPUSSARegisterAllocator::findFreeInRange(
    const TargetRegisterClass *RC,
    SlotIndex Start, SlotIndex End,
    unsigned FreedStart, unsigned FreedEnd,
    const ShadowMap &Shadow) const {
  unsigned CompWidth = TRI->getRegSizeInBits(*RC) / 32;

  for (unsigned Idx = FreedStart; Idx + CompWidth <= FreedEnd; ++Idx) {
    MCRegister BaseReg = AMDGPU::VGPR0 + Idx;
    MCRegister PR = (CompWidth == 1)
        ? BaseReg
        : TRI->getMatchingSuperReg(BaseReg, AMDGPU::sub0, RC);
    if (!PR)
      continue;
    if (isPhysRegFreeForRange(PR, Start, End, Shadow))
      return PR;
  }
  return MCRegister();
}

void AMDGPUSSARegisterAllocator::fillGaps(
    SmallVectorImpl<Gap> &Gaps, unsigned TargetMaxIndex,
    ShadowMap &Shadow, OccupancyPlan &Plan) {
  for (const Gap &G : Gaps) {
    for (auto &E : Shadow) {
      if (!E.VReg.isValid())
        continue;
      unsigned CompIdx = TRI->getHWRegIndex(E.PhysReg);
      if (CompIdx + E.Width <= TargetMaxIndex)
        continue;
      if (E.Width > G.Width)
        continue;
      if (E.Start < G.Start || E.End > G.End)
        continue;

      MCRegister PR = findFreeInRange(MRI->getRegClass(E.VReg),
                                      E.Start, E.End,
                                      G.HWIndex, G.HWIndex + G.Width, Shadow);
      if (PR) {
        Plan.Recolors.push_back({E.VReg, E.PhysReg, PR});
        E.PhysReg = PR;
        LLVM_DEBUG(dbgs() << "    gap-fill: " << printReg(E.VReg, TRI)
                          << " -> " << TRI->getName(PR) << "\n");
      }
    }
  }
}

bool AMDGPUSSARegisterAllocator::planSplit(OccupancyPlan &Plan,
                                           unsigned TargetMaxIndex) {
  ShadowMap Shadow;
  for (const auto &[VReg, PhysReg] : ColorMap) {
    const LiveInterval &LI = LIS->getInterval(VReg);
    unsigned W = TRI->getRegSizeInBits(*MRI->getRegClass(VReg)) / 32;
    Shadow.push_back({VReg, PhysReg, LI.beginIndex(), LI.endIndex(), W});
  }

  SmallVector<Gap> Gaps;

  // Build candidate list: (width, index into Shadow), widest first
  SmallVector<std::pair<unsigned, unsigned>> Candidates;
  for (unsigned I = 0; I < Shadow.size(); ++I) {
    unsigned W = TRI->getRegSizeInBits(*MRI->getRegClass(Shadow[I].VReg));
    Candidates.push_back({W, I});
  }
  llvm::sort(Candidates, [](const auto &A, const auto &B) {
    return A.first > B.first;
  });

  auto computeShadowMaxIndex = [&]() -> unsigned {
    unsigned Max = 0;
    for (const auto &E : Shadow) {
      unsigned Idx = TRI->getHWRegIndex(E.PhysReg);
      Max = std::max(Max, Idx + E.Width);
    }
    return Max;
  };

  LLVM_DEBUG(dbgs() << "  Candidates: " << Candidates.size() << "\n");
  for (const auto &[SplitWidth, SplitIdx] : Candidates) {
    auto &SE = Shadow[SplitIdx];
    unsigned SplitHWStart = TRI->getHWRegIndex(SE.PhysReg);
    unsigned SplitHWEnd = SplitHWStart + SplitWidth / 32;
    LLVM_DEBUG(dbgs() << "  Try split " << printReg(SE.VReg, TRI)
                      << " w=" << SplitWidth
                      << " [" << SE.Start << "," << SE.End << ")\n");

    for (unsigned OI = 0; OI < Shadow.size(); ++OI) {
      if (OI == SplitIdx)
        continue;
      const auto &OE = Shadow[OI];
      if (!OE.VReg.isValid())
        continue;
      if (OE.Width * 32 < SplitWidth)
        continue;

      SlotIndex DeathPoint = OE.End;
      LLVM_DEBUG(dbgs() << "    Other " << printReg(OE.VReg, TRI)
                        << " death=" << DeathPoint << "\n");
      if (DeathPoint <= SE.Start || DeathPoint >= SE.End)
        continue;

      MachineBasicBlock *SplitBB = LIS->getMBBFromIndex(DeathPoint);
      if (MLI->getLoopFor(SplitBB))
        continue;

      if (!isPhysRegFreeForRange(OE.PhysReg, DeathPoint, SE.End, Shadow))
        continue;

      Plan.Splits.push_back({SE.VReg, DeathPoint, OE.PhysReg});
      LLVM_DEBUG(dbgs() << "  Split " << printReg(SE.VReg, TRI)
                        << " at " << DeathPoint
                        << " -> " << TRI->getName(OE.PhysReg) << "\n");

      // Update Shadow: shorten split vreg, add fragment
      SlotIndex OrigEnd = SE.End;
      SE.End = DeathPoint;
      Shadow.push_back({Register(), OE.PhysReg, DeathPoint, OrigEnd, SE.Width});

      // Collect partial-death gaps
      if (OE.VReg.isValid() && LIS->hasInterval(OE.VReg)) {
        unsigned OHWS = TRI->getHWRegIndex(OE.PhysReg);
        for (const LiveInterval::SubRange &SR :
             LIS->getInterval(OE.VReg).subranges()) {
          if (SR.endIndex() == OE.End)
            continue;
          unsigned LO = llvm::countr_zero(SR.LaneMask.getAsInteger());
          unsigned LW = SR.LaneMask.getNumLanes();
          Gaps.push_back({OHWS + LO, LW, SR.endIndex(), OE.End});
        }
      }

      // Re-color competitors into freed slots
      for (auto &CE : Shadow) {
        if (!CE.VReg.isValid() || CE.VReg == SE.VReg)
          continue;
        unsigned CIdx = TRI->getHWRegIndex(CE.PhysReg);
        if (CIdx + CE.Width <= TargetMaxIndex)
          continue;

        MCRegister NewPR = findFreeInRange(
            MRI->getRegClass(CE.VReg), CE.Start, CE.End,
            SplitHWStart, SplitHWEnd, Shadow);
        if (NewPR) {
          Plan.Recolors.push_back({CE.VReg, CE.PhysReg, NewPR});
          CE.PhysReg = NewPR;
          LLVM_DEBUG(dbgs() << "    recolor: " << printReg(CE.VReg, TRI)
                            << " -> " << TRI->getName(NewPR) << "\n");
        }
      }

      Plan.NewMaxIndex = computeShadowMaxIndex();
      if (Plan.NewMaxIndex <= TargetMaxIndex)
        break;
    }

    if (Plan.NewMaxIndex > 0 && Plan.NewMaxIndex <= TargetMaxIndex)
      break;
  }

  if (!Gaps.empty())
    fillGaps(Gaps, TargetMaxIndex, Shadow, Plan);

  Plan.NewMaxIndex = computeShadowMaxIndex();
  return Plan.NewMaxIndex <= TargetMaxIndex;
}

void AMDGPUSSARegisterAllocator::commitPlan(
    MachineFunction &MF, const OccupancyPlan &Plan) {
  MachineLaneSSAUpdater SSAUpdater(MF, *LIS, *MDT, *TRI);

  for (const SplitPlan &SP : Plan.Splits) {
    const TargetRegisterClass *RC = MRI->getRegClass(SP.VReg);
    Register NewVReg = MRI->createVirtualRegister(RC);

    MachineInstr *SplitMI = LIS->getInstructionFromIndex(SP.SplitPoint);
    MachineBasicBlock *SplitBB = SplitMI
        ? SplitMI->getParent()
        : LIS->getMBBFromIndex(SP.SplitPoint);
    MachineBasicBlock::iterator InsertPos = SplitMI
        ? SplitMI->getIterator()
        : SplitBB->getFirstTerminator();

    MachineInstr *CopyMI = BuildMI(*SplitBB, InsertPos, DebugLoc(),
                                   TII->get(TargetOpcode::COPY), NewVReg)
                               .addReg(SP.VReg);
    LIS->InsertMachineInstrInMaps(*CopyMI);

    ColorMap[NewVReg] = SP.NewPhysReg;

    LaneBitmask Mask = MRI->getMaxLaneMaskForVReg(SP.VReg);
    SSAUpdater.rewriteDominatedUses(SP.VReg, NewVReg, Mask);

    LIS->removeInterval(SP.VReg);
    LIS->createAndComputeVirtRegInterval(SP.VReg);
    LIS->createAndComputeVirtRegInterval(NewVReg);

    LLVM_DEBUG(dbgs() << "  Committed split: " << printReg(SP.VReg, TRI)
                      << " -> " << printReg(NewVReg, TRI)
                      << " physreg=" << TRI->getName(SP.NewPhysReg) << "\n");
  }

  for (const RecolorPlan &RP : Plan.Recolors) {
    ColorMap[RP.VReg] = RP.NewPhysReg;
    LLVM_DEBUG(dbgs() << "  Committed recolor: " << printReg(RP.VReg, TRI)
                      << " " << TRI->getName(RP.OldPhysReg)
                      << " -> " << TRI->getName(RP.NewPhysReg) << "\n");
  }

  LLVM_DEBUG(dbgs() << "Split+Assign: " << Plan.Splits.size() << " splits, "
                    << Plan.Recolors.size() << " recolors, "
                    << "newMax=" << Plan.NewMaxIndex << "\n");
}

bool AMDGPUSSARegisterAllocator::splitForOccupancy(MachineFunction &MF) {
  unsigned MaxVGPR = computeMaxPhysRegIndex(/*IsVGPR=*/true);
  unsigned CurrentOcc = ST->getOccupancyWithNumVGPRs(MaxVGPR, 0);
  if (CurrentOcc >= ST->getMaxWavesPerEU())
    return false;

  unsigned TargetOcc = CurrentOcc + 1;
  unsigned TargetMaxVGPR = ST->getMaxNumVGPRs(TargetOcc, /*DynamicVGPRBlockSize=*/0);

  // Early exit: already within target — avoids Shadow copy in planSplit
  if (MaxVGPR <= TargetMaxVGPR)
    return false;

  LLVM_DEBUG(dbgs() << "\n=== Split for occupancy ===\n"
                    << "  Current: " << MaxVGPR << " VGPRs, occ=" << CurrentOcc
                    << "\n  Target: " << TargetMaxVGPR << " VGPRs, occ="
                    << TargetOcc << "\n");

  OccupancyPlan Plan;
  if (!planSplit(Plan, TargetMaxVGPR)) {
    LLVM_DEBUG(dbgs() << "  No viable plan, aborting\n");
    return false;
  }

  commitPlan(MF, Plan);
  return true;
}

// === Main entry point ===

bool AMDGPUSSARegisterAllocator::runOnMachineFunction(MachineFunction &MF) {
  TRI = static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  TII = static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MRI = &MF.getRegInfo();
  ST = &MF.getSubtarget<GCNSubtarget>();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  Indexes = &getAnalysis<SlotIndexesWrapperPass>().getSI();
  RegClassInfo.runOnMachineFunction(MF);

  LLVM_DEBUG(dbgs() << "AMDGPUSSARegisterAllocator: Processing "
                    << MF.getName() << "\n");

  classifyVRegs();
  OccupiedRegUnits.clear();
  OccupiedRegUnits.resize(TRI->getNumRegUnits());
  ColorMap.clear();

  colorAndSplit(MF);

  // TODO: SSA-Destruct (phi to copies/swaps)
  // TODO: Operand rewrite (vreg to physreg)

  return !ColorMap.empty();
}

MachineFunctionPass *llvm::createAMDGPUSSARegisterAllocatorPass() {
  return new AMDGPUSSARegisterAllocator();
}

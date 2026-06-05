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
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-allocator"

char AMDGPUSSARegisterAllocator::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUSSARegisterAllocator, DEBUG_TYPE,
                      "AMDGPU SSA Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUSSARegisterAllocator, DEBUG_TYPE,
                    "AMDGPU SSA Register Allocator", false, false)

// === Coloring ===

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

        const TargetRegisterClass *RC = MRI->getRegClass(VReg);
        unsigned Idx = TRI->getHWRegIndex(Chosen);
        unsigned W = TRI->getRegSizeInBits(*RC) / 32;
        if (TRI->isVGPRClass(RC))
          MaxVGPRIdx = std::max(MaxVGPRIdx, Idx + W);
        else if (TRI->isSGPRClass(RC))
          MaxSGPRIdx = std::max(MaxSGPRIdx, Idx + W);
      }

      if (MI.isPHI())
        continue;

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

void AMDGPUSSARegisterAllocator::color() {
  for (unsigned Width : ColoringOrder)
    colorByWidth(Width);

  LLVM_DEBUG({
    dbgs() << "\nColoring result:\n";
    for (const auto &[VReg, PhysReg] : ColorMap)
      dbgs() << "  " << printReg(VReg, TRI) << " -> "
             << TRI->getName(PhysReg) << "\n";
  });
}

// === SSA Destruction + Operand Rewrite ===

bool AMDGPUSSARegisterAllocator::hasCFPseudos(MachineFunction &MF) const {
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB.terminators())
      switch (MI.getOpcode()) {
      case AMDGPU::SI_IF:
      case AMDGPU::SI_ELSE:
      case AMDGPU::SI_IF_BREAK:
      case AMDGPU::SI_LOOP:
      case AMDGPU::SI_END_CF:
        return true;
      default:
        break;
      }
  return false;
}

void AMDGPUSSARegisterAllocator::emitSwap(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    MCRegister RegA, MCRegister RegB) {
  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(RegA);
  unsigned RegWidth = TRI->getRegSizeInBits(*RC);

  if (RegWidth <= 32) {
    if (ST->hasSwap()) {
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::V_SWAP_B32), RegA)
          .addDef(RegB)
          .addReg(RegB)
          .addReg(RegA);
    } else {
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::V_XOR_B32_e64),
              RegA)
          .addReg(RegA).addReg(RegB);
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::V_XOR_B32_e64),
              RegB)
          .addReg(RegA).addReg(RegB);
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(AMDGPU::V_XOR_B32_e64),
              RegA)
          .addReg(RegA).addReg(RegB);
    }
    return;
  }

  const unsigned DWordBytes = 4;
  ArrayRef<int16_t> Parts = TRI->getRegSplitParts(RC, DWordBytes);
  for (int16_t SubIdx : Parts) {
    MCRegister SubA = TRI->getSubReg(RegA, SubIdx);
    MCRegister SubB = TRI->getSubReg(RegB, SubIdx);
    emitSwap(MBB, InsertPt, SubA, SubB);
  }
}

void AMDGPUSSARegisterAllocator::resolvePermutation(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    SmallVectorImpl<std::pair<MCRegister, MCRegister>> &Copies, bool IsVGPR) {
  if (Copies.empty())
    return;

  DenseMap<MCRegister, MCRegister> DstToSrc;
  DenseMap<MCRegister, unsigned> SrcRefCount;
  for (auto &[Src, Dst] : Copies) {
    DstToSrc[Dst] = Src;
    SrcRefCount[Src]++;
  }

  // Phase 1: emit chain copies via worklist.
  // Seed with all destinations that are not sources of any remaining copy.
  SmallVector<MCRegister> Ready;
  for (auto &[Dst, Src] : DstToSrc)
    if (SrcRefCount[Dst] == 0)
      Ready.push_back(Dst);

  while (!Ready.empty()) {
    MCRegister Dst = Ready.pop_back_val();
    MCRegister Src = DstToSrc[Dst];
    DstToSrc.erase(Dst);
    BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY), Dst)
        .addReg(Src);
    LLVM_DEBUG(dbgs() << "    copy: " << TRI->getName(Src) << " -> "
                      << TRI->getName(Dst) << "\n");
    if (--SrcRefCount[Src] == 0 && DstToSrc.count(Src))
      Ready.push_back(Src);
  }

  // Phase 2: all remaining entries form cycles (chains were drained above).
  unsigned &MaxIdx = IsVGPR ? MaxVGPRIdx : MaxSGPRIdx;
  const MachineFunction &MF = *MBB.getParent();
  unsigned MaxHWLimit = IsVGPR
      ? ST->getMaxNumVGPRs(MF)
      : ST->getMaxNumSGPRs(MF);
  unsigned CurrentOcc = IsVGPR
      ? ST->getOccupancyWithNumVGPRs(MaxIdx, DynVGPRBlockSize)
      : ST->getOccupancyWithNumSGPRs(MaxIdx);

  while (!DstToSrc.empty()) {
    // Pick any entry as cycle start — all remaining entries form disjoint
    // cycles, and the walk traces the full cycle regardless of entry point.
    MCRegister CycleStart = DstToSrc.begin()->first;

    // Tier 1: scratch register if it doesn't reduce occupancy.
    // Scratch must match the cycle's register width.
    unsigned CycleWidth =
        TRI->getRegSizeInBits(*TRI->getPhysRegBaseClass(CycleStart)) / 32;
    unsigned ScratchOcc = IsVGPR
        ? ST->getOccupancyWithNumVGPRs(MaxIdx + CycleWidth, DynVGPRBlockSize)
        : ST->getOccupancyWithNumSGPRs(MaxIdx + CycleWidth);

    if (ScratchOcc == CurrentOcc && MaxIdx + CycleWidth <= MaxHWLimit) {
      MCRegister ScratchBase = IsVGPR
          ? MCRegister(AMDGPU::VGPR0 + MaxIdx)
          : MCRegister(AMDGPU::SGPR0 + MaxIdx);
      MCRegister Scratch = (CycleWidth == 1)
          ? ScratchBase
          : TRI->getMatchingSuperReg(ScratchBase, AMDGPU::sub0,
                TRI->getPhysRegBaseClass(CycleStart));
      MaxIdx += CycleWidth;

      CurrentOcc = ScratchOcc;

      LLVM_DEBUG(dbgs() << "    cycle via scratch "
                        << TRI->getName(Scratch) << ":\n");

      // Save CycleStart — it will be overwritten by the first copy.
      // The last register in the walk receives this saved value.
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY), Scratch)
          .addReg(CycleStart);
      LLVM_DEBUG(dbgs() << "      save: " << TRI->getName(CycleStart)
                        << " -> " << TRI->getName(Scratch) << "\n");

      MCRegister Cur = CycleStart;
      while (true) {
        MCRegister Src = DstToSrc[Cur];
        DstToSrc.erase(Cur);
        if (!DstToSrc.count(Src)) {
          assert(Src == CycleStart && "Cycle walk did not return to start");
          BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY), Cur)
              .addReg(Scratch);
          LLVM_DEBUG(dbgs() << "      restore: " << TRI->getName(Scratch)
                            << " -> " << TRI->getName(Cur) << "\n");
          break;
        }
        BuildMI(MBB, InsertPt, DebugLoc(), TII->get(TargetOpcode::COPY), Cur)
            .addReg(Src);
        LLVM_DEBUG(dbgs() << "      " << TRI->getName(Src) << " -> "
                          << TRI->getName(Cur) << "\n");
        Cur = Src;
      }
      continue;
    }

    // Tier 2/3: break cycle pairwise with V_SWAP_B32 (GFX9+) or XOR.
    // Collect the full cycle, then emit n-1 swaps from tail to head.
    LLVM_DEBUG(dbgs() << "    cycle via "
                      << (ST->hasSwap() ? "V_SWAP_B32" : "XOR") << ":\n");
    SmallVector<MCRegister> Cycle;
    MCRegister Cur = CycleStart;
    while (DstToSrc.count(Cur)) {
      Cycle.push_back(Cur);
      MCRegister Next = DstToSrc[Cur];
      DstToSrc.erase(Cur);
      Cur = Next;
    }
    for (int I = Cycle.size() - 1; I > 0; --I) {
      emitSwap(MBB, InsertPt, Cycle[I - 1], Cycle[I]);
      LLVM_DEBUG(dbgs() << "      swap " << TRI->getName(Cycle[I - 1])
                        << " <-> " << TRI->getName(Cycle[I]) << "\n");
    }
  }
}

void AMDGPUSSARegisterAllocator::lowerPHIs(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "\n=== SSA Destruction ===\n");

  SmallVector<MachineInstr *, 16> PHIsToErase;

  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty() || !MBB.front().isPHI())
      continue;

    DenseMap<MachineBasicBlock *,
             SmallVector<std::pair<MCRegister, MCRegister>>>
        PredCopies;

    // All PHIs in one block share the same register file.
    Register FirstDst = MBB.front().getOperand(0).getReg();
    bool IsVGPR = TRI->isVGPRClass(MRI->getRegClass(FirstDst));

    for (MachineInstr &MI : MBB) {
      if (!MI.isPHI())
        break;

      Register DstVReg = MI.getOperand(0).getReg();
      MCRegister DstPhys = ColorMap.lookup(DstVReg);
      assert(DstPhys && "PHI result not colored");

      for (unsigned I = 1, E = MI.getNumOperands(); I < E; I += 2) {
        Register SrcVReg = MI.getOperand(I).getReg();
        MachineBasicBlock *Pred = MI.getOperand(I + 1).getMBB();
        MCRegister SrcPhys = ColorMap.lookup(SrcVReg);
        assert(SrcPhys && "PHI source not colored");

        if (SrcPhys != DstPhys)
          PredCopies[Pred].push_back({SrcPhys, DstPhys});
      }

      PHIsToErase.push_back(&MI);
    }

    for (auto &[Pred, Copies] : PredCopies) {
      MachineBasicBlock *InsertMBB = Pred;
      if (Pred->succ_size() > 1 && MBB.pred_size() > 1) {
        LLVM_DEBUG(dbgs() << "  Splitting critical edge "
                          << printMBBReference(*Pred) << " -> "
                          << printMBBReference(MBB) << "\n");
        InsertMBB = Pred->SplitCriticalEdge(&MBB, *this);
        assert(InsertMBB && "Failed to split critical edge");
      }

      LLVM_DEBUG(dbgs() << "  Edge " << printMBBReference(*InsertMBB) << " -> "
                        << printMBBReference(MBB) << ":\n");
      auto InsertPt = InsertMBB->getFirstTerminator();
      resolvePermutation(*InsertMBB, InsertPt, Copies, IsVGPR);
    }
  }

  for (MachineInstr *PHI : PHIsToErase)
    PHI->eraseFromParent();

  LLVM_DEBUG(dbgs() << "  Erased " << PHIsToErase.size() << " PHIs\n");
}

void AMDGPUSSARegisterAllocator::rewriteOperands(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "\n=== Operand Rewrite ===\n");

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;

        Register VReg = MO.getReg();
        MCRegister PhysReg = ColorMap.lookup(VReg);
        assert(PhysReg && "Virtual register not colored");

        unsigned SubIdx = MO.getSubReg();
        if (SubIdx) {
          PhysReg = TRI->getSubReg(PhysReg, SubIdx);
          assert(PhysReg && "Invalid subreg index");
          MO.setSubReg(0);
        }
        MO.setReg(PhysReg);
      }
    }
  }
}

void AMDGPUSSARegisterAllocator::destroySSAAndRewrite(MachineFunction &MF) {
  if (hasCFPseudos(MF)) {
    LLVM_DEBUG(dbgs() << "SSA Destruction: skipped — "
                      "SI control-flow pseudos present\n");
    return;
  }

  lowerPHIs(MF);
  rewriteOperands(MF);

  MRI->leaveSSA();
  MRI->invalidateLiveness();
}

// === Main entry point ===

bool AMDGPUSSARegisterAllocator::runOnMachineFunction(MachineFunction &MF) {
  TRI = static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  TII = static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MRI = &MF.getRegInfo();
  ST = &MF.getSubtarget<GCNSubtarget>();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  RegClassInfo.runOnMachineFunction(MF);
  DynVGPRBlockSize = ST->isDynamicVGPREnabled()
      ? ST->getDynamicVGPRBlockSize() : 0;

  LLVM_DEBUG(dbgs() << "AMDGPUSSARegisterAllocator: Processing "
                    << MF.getName() << "\n");

  classifyVRegs();
  OccupiedRegUnits.clear();
  OccupiedRegUnits.resize(TRI->getNumRegUnits());
  ColorMap.clear();
  MaxVGPRIdx = 0;
  MaxSGPRIdx = 0;

  color();
  destroySSAAndRewrite(MF);

  return true;
}

MachineFunctionPass *llvm::createAMDGPUSSARegisterAllocatorPass() {
  return new AMDGPUSSARegisterAllocator();
}

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
    const TargetRegisterClass *RC, const LiveInterval &VI,
    ArrayRef<std::pair<MCRegister, const LiveInterval *>> WiderDefs) {
  LLVM_DEBUG({
    dbgs() << "    Allocation order for " << TRI->getRegClassName(RC) << ":";
    for (MCRegister PR : RegClassInfo.getOrder(RC))
      dbgs() << " " << TRI->getName(PR);
    dbgs() << "\n";
  });

  // Augment OccupiedRegUnits with wider-width assignments that overlap VI.
  // Two sources: (1) WiderDefs — wider defs in THIS block not yet live at
  // BBStart (O(k), k = wider defs in block); (2) ColorMap scan for wider
  // cross-block entries (O(|ColorMap|), but only in narrower width passes
  // after wider passes committed their assignments).
  unsigned VIWidth = TRI->getRegSizeInBits(*RC);
  BitVector OccupiedAtDef = OccupiedRegUnits;
  for (const auto &[WPhysReg, WLI] : WiderDefs) {
    if (WLI->overlaps(VI)) {
      for (MCRegUnit Unit : TRI->regunits(WPhysReg))
        OccupiedAtDef.set(Unit);
    }
  }
  for (const auto &[WReg, WPhysReg] : ColorMap) {
    if (TRI->getRegSizeInBits(*MRI->getRegClass(WReg)) <= VIWidth)
      continue;
    if (LIS->getInterval(WReg).overlaps(VI)) {
      for (MCRegUnit Unit : TRI->regunits(WPhysReg))
        OccupiedAtDef.set(Unit);
    }
  }

  for (MCRegister PR : RegClassInfo.getOrder(RC)) {
    bool Free = true;
    for (MCRegUnit Unit : TRI->regunits(PR))
      if (OccupiedAtDef.test(Unit)) { Free = false; break; }
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

void AMDGPUSSARegisterAllocator::color() {
  LLVM_DEBUG({
    dbgs() << "Coloring order (width descending):";
    for (unsigned W : ColoringOrder)
      dbgs() << " " << W;
    dbgs() << "\n";
  });

  // Function-wide width-descending: color ALL defs of the widest width across
  // all blocks before any narrower width. This prevents narrow defs from
  // fragmenting alignment slots needed by wider tuples (e.g., a VGPR_32 at an
  // odd index blocking an even-aligned VReg_64 pair on gfx90a).
  //
  // Wider assignments are committed to ColorMap before narrower passes start,
  // so seedOccupiedAtBBEntry naturally catches cross-block wider live-ins.
  // For wider defs born mid-block (not live at BBStart), a per-block WiderDefs
  // pre-scan collects them from ColorMap — O(|block|), same cost as the walk.
  for (unsigned Width : ColoringOrder) {
    for (auto *Node : depth_first(MDT->getRootNode())) {
      MachineBasicBlock *MBB = Node->getBlock();

      LLVM_DEBUG(dbgs() << "\n=== Width pass: " << Width << "-bit, "
                        << printMBBReference(*MBB) << " ===\n");

      // Pre-scan: collect wider defs in THIS block from prior width passes.
      // These are defs not live at BBStart (born mid-block) whose physregs
      // must be avoided by the current narrower pass via LI.overlaps().
      SmallVector<std::pair<MCRegister, const LiveInterval *>, 8> WiderDefs;
      for (MachineInstr &MI : *MBB)
        for (MachineOperand &MO : MI.defs())
          if (MO.isReg() && MO.getReg().isVirtual()) {
            Register Reg = MO.getReg();
            if (TRI->getRegSizeInBits(*MRI->getRegClass(Reg)) > Width)
              if (auto It = ColorMap.find(Reg); It != ColorMap.end())
                WiderDefs.push_back(
                    {It->second, &LIS->getInterval(Reg)});
          }

      seedOccupiedAtBBEntry(MBB);

      for (MachineInstr &MI : *MBB) {
        // Kill uses before coloring defs: a def can reuse the physreg of
        // a source that dies at this instruction (no interference without
        // early-clobber). PHIs skipped: their sources are live only to
        // predecessor boundaries, and markFree would clear physregs that
        // preceding PHI defs already claimed.
        if (!MI.isPHI()) {
          SlotIndex NextSI = LIS->getInstructionIndex(MI).getRegSlot().getNextSlot();
          for (const MachineOperand &MO : MI.uses()) {
            if (!MO.isReg())
              continue;
            Register Reg = MO.getReg();
            if (Reg.isPhysical()) {
              for (MCRegUnit Unit : TRI->regunits(Reg))
                if (!LIS->getRegUnit(Unit).liveAt(NextSI))
                  OccupiedRegUnits.reset(Unit);
              continue;
            }
            auto It = ColorMap.find(Reg);
            if (It == ColorMap.end())
              continue;
            if (!LIS->getInterval(Reg).liveAt(NextSI)) {
              markFree(It->second);
              LLVM_DEBUG(dbgs() << "    kill: " << printReg(Reg, TRI)
                                << " free " << TRI->getName(It->second) << "\n");
            }
          }
        }

        for (MachineOperand &MO : MI.defs()) {
          Register Reg = MO.getReg();
          if (!Reg.isVirtual()) {
            markOccupied(Reg);
            continue;
          }

          if (TRI->getRegSizeInBits(*MRI->getRegClass(Reg)) != Width) {
            if (auto It = ColorMap.find(Reg); It != ColorMap.end()) {
              markOccupied(It->second);
              LLVM_DEBUG(dbgs() << "    mark wider def: "
                                << printReg(Reg, TRI) << " -> "
                                << TRI->getName(It->second) << "\n");
            }
            continue;
          }

          MCRegister Chosen;
          unsigned UseOpIdx;
          if (MI.isRegTiedToUseOperand(MO.getOperandNo(), &UseOpIdx)) {
            Chosen = ColorMap.lookup(MI.getOperand(UseOpIdx).getReg());
            assert(Chosen && "Tied use must be colored already");
            LLVM_DEBUG(dbgs() << "    tied: " << printReg(Reg, TRI)
                              << " inherits " << TRI->getName(Chosen) << "\n");
          } else {
            Chosen = pickFreePhysReg(MRI->getRegClass(Reg),
                                     LIS->getInterval(Reg), WiderDefs);
            assert(Chosen && "Failed to find free physreg");
            LLVM_DEBUG(dbgs() << "    color: " << printReg(Reg, TRI)
                              << " -> " << TRI->getName(Chosen) << "\n");
          }

          ColorMap[Reg] = Chosen;
          markOccupied(Chosen);

          const TargetRegisterClass *RC = MRI->getRegClass(Reg);
          unsigned Idx = TRI->getHWRegIndex(Chosen);
          unsigned W = TRI->getRegSizeInBits(*RC) / 32;
          if (TRI->isVGPRClass(RC))
            MaxVGPRIdx = std::max(MaxVGPRIdx, Idx + W);
          else if (TRI->isSGPRClass(RC))
            MaxSGPRIdx = std::max(MaxSGPRIdx, Idx + W);
        }
      }
    }
  }

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
    SmallVectorImpl<std::pair<MCRegister, MCRegister>> &Copies) {
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
  // A permutation cycle is always confined to one register file — a VGPR
  // destination can never equal an SGPR source — so the file (and thus the
  // scratch counter, HW limit, occupancy model and swap lowering) is derived
  // per cycle from its own registers, never from a block-wide assumption.
  const MachineFunction &MF = *MBB.getParent();

  while (!DstToSrc.empty()) {
    // Pick any entry as cycle start — all remaining entries form disjoint
    // cycles, and the walk traces the full cycle regardless of entry point.
    MCRegister CycleStart = DstToSrc.begin()->first;

    bool IsVGPR = TRI->isVGPRClass(TRI->getPhysRegBaseClass(CycleStart));
    unsigned &MaxIdx = IsVGPR ? MaxVGPRIdx : MaxSGPRIdx;
    unsigned MaxHWLimit =
        IsVGPR ? ST->getMaxNumVGPRs(MF) : ST->getMaxNumSGPRs(MF);
    unsigned CurrentOcc =
        IsVGPR ? ST->getOccupancyWithNumVGPRs(MaxIdx, DynVGPRBlockSize)
               : ST->getOccupancyWithNumSGPRs(MaxIdx);

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

    for (MachineInstr &MI : MBB) {
      if (!MI.isPHI())
        break;

      Register DstVReg = MI.getOperand(0).getReg();
      MCRegister DstPhys = ColorMap.lookup(DstVReg);
      assert(DstPhys && "PHI result not colored");

      // The PHI result physreg flows into this block from each predecessor.
      // After the PHI is erased, the block has no definition of DstPhys, so
      // we must declare it as a live-in so the verifier recognises it.
      if (!MBB.isLiveIn(DstPhys))
        MBB.addLiveIn(DstPhys);

      for (unsigned I = 1, E = MI.getNumOperands(); I < E; I += 2) {
        Register SrcVReg = MI.getOperand(I).getReg();
        MachineBasicBlock *Pred = MI.getOperand(I + 1).getMBB();
        MCRegister SrcPhys = ColorMap.lookup(SrcVReg);
        assert(SrcPhys && "PHI source not colored");

        // A PHI source may name a subregister (e.g. %x.sub0). The copy must
        // move the corresponding sub-physreg, not the full tuple, otherwise we
        // emit an illegal width-mismatched copy.
        if (unsigned SubIdx = MI.getOperand(I).getSubReg()) {
          SrcPhys = TRI->getSubReg(SrcPhys, SubIdx);
          assert(SrcPhys && "Invalid subreg index on PHI source");
        }

        if (SrcPhys != DstPhys)
          PredCopies[Pred].push_back({SrcPhys, DstPhys});
      }

      PHIsToErase.push_back(&MI);
    }
    MBB.sortUniqueLiveIns();

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
      resolvePermutation(*InsertMBB, InsertPt, Copies);
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

// Update MBB live-in sets with the physical registers assigned to virtual
// registers that are live at each block entry. VirtRegRewriter does this in
// the greedy RA path; without it the machine verifier reports "Using an
// undefined physical register" for cross-block physreg uses.
void AMDGPUSSARegisterAllocator::addPhysRegLiveIns(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    SlotIndex BBStart = LIS->getMBBStartIdx(&MBB);
    for (const auto &[VReg, PhysReg] : ColorMap) {
      if (LIS->getInterval(VReg).liveAt(BBStart)) {
        if (!MBB.isLiveIn(PhysReg))
          MBB.addLiveIn(PhysReg);
      }
    }
    MBB.sortUniqueLiveIns();
  }
}

// Set all MachineFunction properties that downstream passes require after
// SSA destruction and physical register assignment are complete.
// Mirrors the state produced by VirtRegRewriter in the greedy RA path:
//   NoPHIs     — all PHI instructions removed by lowerPHIs()
//   NoVRegs    — all virtual registers replaced with physregs by rewriteOperands()
//   IsSSA      — cleared by leaveSSA() (not SSA anymore)
// TracksLiveness is deliberately preserved: MBB live-in sets contain only
// physregs and remain valid after the rewrite; clearing it would break
// post-RA passes such as MachineLICM that call livein_begin().
void AMDGPUSSARegisterAllocator::finalizeProperties(MachineFunction &MF) {
  MRI->leaveSSA();
  // Remove all virtual register declarations from MRI so that the verifier's
  // NoVRegs check (MRI->getNumVirtRegs() == 0) passes. VirtRegRewriter does
  // the same in the greedy RA path. Instruction operands are already physical
  // after rewriteOperands(); this only removes the stale vreg table entries.
  MRI->clearVirtRegs();
  MF.getProperties().set(MachineFunctionProperties::Property::NoPHIs);
  MF.getProperties().set(MachineFunctionProperties::Property::NoVRegs);
}

// Eliminate REG_SEQUENCE instructions after physreg assignment.
// In the greedy RA path, VirtRegRewriter handles this. We skip VirtRegRewriter,
// so REG_SEQUENCEs that survived into post-RA MIR must be lowered here.
//
// A REG_SEQUENCE:  dst = REG_SEQUENCE src0, sub0, src1, sub1, ...
// is "trivial" if for every (src_i, sub_i): src_i == TRI->getSubReg(dst, sub_i).
// Trivial ones are deleted. Non-trivial ones are lowered to COPY instructions
// placed immediately before the REG_SEQUENCE, then the REG_SEQUENCE is deleted.
void AMDGPUSSARegisterAllocator::eliminateRegSequences(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (!MI.isRegSequence())
        continue;

      MCRegister Dst = MI.getOperand(0).getReg().asMCReg();
      LLVM_DEBUG(dbgs() << "  [RegSeq] lowering " << MI);

      // A REG_SEQUENCE is a *parallel* assignment: all sources are read, then
      // each is written to its destination slice. Collect the non-trivial
      // (Src -> dst-slice) pairs and hand them to resolvePermutation, which
      // sequences them to respect write-after-read hazards (a slice that
      // overwrites a register another pair still needs) and cycles. Emitting
      // the copies naively in operand order corrupts such overlaps.
      SmallVector<std::pair<MCRegister, MCRegister>, 4> Copies;
      for (unsigned I = 1, E = MI.getNumOperands(); I < E; I += 2) {
        MCRegister Src = MI.getOperand(I).getReg().asMCReg();
        unsigned SubIdx = MI.getOperand(I + 1).getImm();
        // The source class may be wider than the slice it fills (e.g. a 64-bit
        // value held in an sgpr_128 vreg). The slice index then also names the
        // matching sub-register of the source — narrow Src to it so the COPY is
        // width-correct. When Src already matches the slice width, SubIdx names
        // no sub-register of Src and getSubReg() returns 0, leaving Src as-is.
        if (MCRegister SubSrc = TRI->getSubReg(Src, SubIdx))
          Src = SubSrc;
        MCRegister Expected = TRI->getSubReg(Dst, SubIdx);
        if (Src != Expected)
          Copies.push_back({Src, Expected});
      }
      resolvePermutation(MBB, MI, Copies);
      MI.eraseFromParent();
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
  eliminateRegSequences(MF);
  addPhysRegLiveIns(MF);
  finalizeProperties(MF);
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

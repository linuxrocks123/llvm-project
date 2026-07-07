//===- MachineLaneSSAUpdater.cpp - SSA repair for Machine IR (lane-aware) ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the MachineLaneSSAUpdater - a universal SSA repair utility
// for Machine IR that handles both regular new definitions and reload-after-
// spill scenarios with full subregister lane awareness.
//
// Key features:
//  - Two explicit entry points:
//    * repairSSAForNewDef - Common use case: caller creates instruction
//    defining
//      existing vreg (violating SSA), updater creates new vreg and repairs
//    * addDefAndRepairAfterSpill - Spill/reload use case: caller creates
//    instruction
//      with new vreg, updater repairs SSA using spill-time EndPoints
//  - Lane-aware PHI insertion with per-edge masks
//  - Pruned IDF computation (NewDefBlocks ∩ LiveIn(OldVR))
//  - Precise LiveInterval extension using captured EndPoints
//  - REG_SEQUENCE insertion only when necessary
//  - Preservation of undef/dead flags on partial definitions
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineLaneSSAUpdater.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "machine-lane-ssa-updater"

using namespace llvm;

//===----------------------------------------------------------------------===//
// MachineLaneSSAUpdater Implementation
//===----------------------------------------------------------------------===//

Register MachineLaneSSAUpdater::repairSSAForNewDef(
    MachineInstr &NewDefMI, Register OrigVReg,
    SmallVectorImpl<MachineOperand *> &PHIRegDefOps) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::repairSSAForNewDef VReg="
                    << OrigVReg << "\n");

  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Scope the rename map to one OrigVReg. The driver processes all defs of an
  // OrigVReg before moving on, so clearing on change is sufficient.
  if (RenameSessionOrig != OrigVReg) {
    RenameSessionOrig = OrigVReg;
    DefInstrToRenamed.clear();
    LanePHIs.clear();

    // Freeze a deep copy of OrigVReg's original LiveInterval as the
    // reaching-def oracle for this session. Must be taken BEFORE any def is
    // renamed, since renaming (in place) removes that def's VNInfo from the
    // live OrigVReg interval when it is later recomputed. The copy preserves
    // each VNInfo's def SlotIndex and isPHIDef flag, which is all reaching
    // resolution needs. Destroy the previous frozen interval BEFORE reclaiming
    // its backing allocator: its subranges live in FrozenAlloc, so resetting
    // the allocator first makes the old interval's destructor double-free them.
    FrozenOrigLI.reset();
    FrozenAlloc.Reset();
    FrozenOrigLI = std::make_unique<LiveInterval>(OrigVReg, 0.0f);
    if (LIS.hasInterval(OrigVReg)) {
      LiveInterval &Src = LIS.getInterval(OrigVReg);
      FrozenOrigLI->assign(Src, FrozenAlloc);
      for (const LiveInterval::SubRange &S : Src.subranges())
        FrozenOrigLI->createSubRangeFrom(FrozenAlloc, S.LaneMask, S);
    }
  }

  // Step 1: Find the def operand for OrigVReg
  MachineOperand *DefOp = nullptr;
  unsigned DefOpIdx = 0;

  for (MachineOperand &MO : NewDefMI.defs()) {
    if (MO.getReg() == OrigVReg) {
      DefOp = &MO;
      break;
    }
    ++DefOpIdx;
  }

  assert(DefOp && "NewDefMI should have a def operand for OrigVReg");
  assert(DefOp->isDef() && "Found operand should be a definition");

  // Step 2: Derive DefMask from the operand's subreg index (if any)
  unsigned SubRegIdx = DefOp->getSubReg();
  LaneBitmask DefMask;

  if (SubRegIdx) {
    // Partial register definition - get lane mask for this subreg
    DefMask = TRI.getSubRegIndexLaneMask(SubRegIdx);
    LLVM_DEBUG(dbgs() << "  Partial def with subreg "
                      << TRI.getSubRegIndexName(SubRegIdx)
                      << ", DefMask=" << PrintLaneMask(DefMask) << "\n");
  } else {
    // Full register definition - get all lanes for this register class
    DefMask = MRI.getMaxLaneMaskForVReg(OrigVReg);
    LLVM_DEBUG(dbgs() << "  Full register def, DefMask="
                      << PrintLaneMask(DefMask) << "\n");
  }

  // Step 3: Create a new virtual register with class matching MaskToRewrite
  // width
  const TargetRegisterClass *RC;
  if (SubRegIdx) {
    // For subreg defs, create register with class for the subreg (narrower)
    const TargetRegisterClass *OrigRC = MRI.getRegClass(OrigVReg);
    RC = TRI.getSubRegisterClass(OrigRC, SubRegIdx);
    assert(RC && "Failed to get subregister class");
  } else {
    // For full register defs, use same class as OrigVReg
    RC = MRI.getRegClass(OrigVReg);
  }
  Register NewSSAVReg = MRI.createVirtualRegister(RC);
  LLVM_DEBUG(dbgs() << "  Created new SSA vreg " << NewSSAVReg
                    << " with RC=" << TRI.getRegClassName(RC) << "\n");

  // Step 4: Replace the operand in NewDefMI to define the new vreg
  // Clear subreg index since NewSSAVReg is a full register of the (possibly
  // narrower) class
  DefOp->setReg(NewSSAVReg);
  if (SubRegIdx) {
    DefOp->setSubReg(
        0); // Clear subreg - NewSSAVReg is full register of narrower class
    DefOp->setIsUndef(
        false); // Clear undef flag - verifier requires undef only with subreg
    LLVM_DEBUG(dbgs() << "  Replaced operand: " << OrigVReg << "."
                      << TRI.getSubRegIndexName(SubRegIdx) << " -> "
                      << NewSSAVReg << " (full narrower register)\n");
  } else {
    LLVM_DEBUG(dbgs() << "  Replaced operand: " << OrigVReg << " -> "
                      << NewSSAVReg << "\n");
  }

  // Step 5: Index the new instruction in SlotIndexes/LIS
  indexNewInstr(NewDefMI);

  // Make this rename available to later PHI construction in this session:
  // record the renamed def by its instruction (with the OrigVReg lanes it
  // covers), so reaching-VNInfo lookups can map a reaching real def to its
  // renamed vreg and rebase target lanes.
  DefInstrToRenamed[&NewDefMI] = {NewSSAVReg, DefMask};

  // Step 6: Perform common SSA repair (PHI placement + use rewriting)
  // LiveInterval for NewSSAVReg will be created by getInterval() as needed
  PHIRegDefOps =
      performSSARepair(NewSSAVReg, OrigVReg, DefMask, NewDefMI.getParent());

  // Step 7: If SSA repair created subregister uses of OrigVReg (e.g., in PHIs
  // or REG_SEQUENCEs), recompute its LiveInterval to create subranges
  LaneBitmask AllLanes = MRI.getMaxLaneMaskForVReg(OrigVReg);
  if (DefMask != AllLanes) {
    LiveInterval &OrigLI = LIS.getInterval(OrigVReg);
    if (!OrigLI.hasSubRanges()) {
      // Check if any uses now access OrigVReg with subregister indices
      bool HasSubregUses = false;
      for (const MachineOperand &MO : MRI.use_operands(OrigVReg)) {
        if (MO.getSubReg() != 0) {
          HasSubregUses = true;
          break;
        }
      }

      if (HasSubregUses) {
        LLVM_DEBUG(dbgs() << "  Recomputing LiveInterval for " << OrigVReg
                          << " after SSA repair created subregister uses\n");
        LIS.removeInterval(OrigVReg);
        LIS.createAndComputeVirtRegInterval(OrigVReg);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "  repairSSAForNewDef complete, returning "
                    << printReg(NewSSAVReg, &TRI) << "\n");
  return NewSSAVReg;
}

//===----------------------------------------------------------------------===//
// Common SSA Repair Logic
//===----------------------------------------------------------------------===//

SmallVector<MachineOperand *>
MachineLaneSSAUpdater::performSSARepair(Register NewVReg, Register OrigVReg,
                                        LaneBitmask DefMask,
                                        MachineBasicBlock *DefBB) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::performSSARepair NewVReg="
                    << NewVReg << " OrigVReg=" << OrigVReg
                    << " DefMask=" << PrintLaneMask(DefMask) << "\n");

  // Step 1: Use worklist-driven PHI placement. Each PHI is paired with the
  // OrigVReg lane it covers.
  SmallVector<std::pair<MachineOperand *, LaneBitmask>> AllPHIVResults =
      insertLaneAwarePHI(OrigVReg, DefMask);

  // Step 2: Rewrite dominated uses once for each new register. Each PHI is
  // rewritten with ITS OWN lane (per-subrange in the reaching path; DefMask in
  // the legacy path) so subreg rebasing is correct.
  // Note: getInterval() will automatically create LiveIntervals if needed
  rewriteDominatedUses(OrigVReg, NewVReg, DefMask);
  for (auto &[PHIVRes, Lane] : AllPHIVResults)
    rewriteDominatedUses(OrigVReg, PHIVRes->getReg(), Lane);

  // Step 3: Renumber values if needed
  LiveInterval &NewLI = LIS.getInterval(NewVReg);
  NewLI.RenumberValues();

  // Also renumber PHI intervals
  for (auto &[PHIVRes, Lane] : AllPHIVResults) {
    LiveInterval &PHILI = LIS.getInterval(PHIVRes->getReg());
    PHILI.RenumberValues();
  }

  // Recompute OrigVReg's LiveInterval to account for PHI operands
  // We do a full recomputation because PHI operands may reference subregisters
  // that weren't previously live on those paths, and we need to extend liveness
  // from the definition to the PHI use.
  LIS.removeInterval(OrigVReg);
  LIS.createAndComputeVirtRegInterval(OrigVReg);

  // Note: We do NOT call shrinkToUses on OrigVReg even after recomputation
  // because: shrinkToUses has a fundamental bug with PHI operands - it doesn't
  // understand that PHI operands require their source lanes to be live at the
  // END of predecessor blocks. When it sees a PHI operand like "%0.sub2_sub3"
  // from BB3, it only considers the PHI location (start of join block), not the
  // predecessor end where the value must be available. This causes it to
  // incorrectly shrink away lanes that ARE needed by PHI operands, leading to
  // verification errors: "Not all lanes of PHI source live at use". The
  // createAndComputeVirtRegInterval already produces correct, minimal liveness
  // that includes PHI uses properly.

  // Step 4: Update operand flags to match the LiveIntervals
  updateDeadFlags(NewVReg);
  for (auto &[PHIVRes, Lane] : AllPHIVResults)
    updateDeadFlags(PHIVRes->getReg());

  LLVM_DEBUG(dbgs() << "  performSSARepair complete\n");
  SmallVector<MachineOperand *> Results;
  for (auto &[PHIVRes, Lane] : AllPHIVResults)
    Results.push_back(PHIVRes);
  return Results;
}

//===----------------------------------------------------------------------===//
// Internal Helper Methods (Stubs)
//===----------------------------------------------------------------------===//

SlotIndex MachineLaneSSAUpdater::indexNewInstr(MachineInstr &MI) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::indexNewInstr: " << MI);

  // Register the instruction in SlotIndexes and LiveIntervals
  // This is typically done automatically when instructions are inserted,
  // but we need to ensure it's properly indexed
  SlotIndexes *SI = LIS.getSlotIndexes();

  // Check if instruction is already indexed
  if (SI->hasIndex(MI)) {
    SlotIndex Idx = SI->getInstructionIndex(MI);
    LLVM_DEBUG(dbgs() << "  Already indexed at " << Idx << "\n");
    return Idx;
  }

  // Insert the instruction in maps - this should be done by the caller
  // before calling our SSA repair methods, but we can verify
  LIS.InsertMachineInstrInMaps(MI);

  SlotIndex Idx = SI->getInstructionIndex(MI);
  LLVM_DEBUG(dbgs() << "  Indexed at " << Idx << "\n");
  return Idx;
}

SmallVector<std::pair<MachineOperand *, LaneBitmask>>
MachineLaneSSAUpdater::insertLaneAwarePHI(Register OrigVReg,
                                          LaneBitmask DefMask) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::insertLaneAwarePHI OrigVReg="
                    << OrigVReg << " DefMask=" << PrintLaneMask(DefMask)
                    << "\n");

  SmallVector<std::pair<MachineOperand *, LaneBitmask>> AllCreatedPHIs;

  // PHIs go exactly at the join points already recorded in OrigVReg's frozen
  // interval as PHI-def VNInfos. LiveIntervalCalc computed those when the
  // interval was built, so per lane they ARE the pruned iterated dominance
  // frontier -- no IDF recomputation needed. Dead lanes (no merge) have no
  // PHI-def VNInfo, so they correctly get no PHI. Operands are resolved from
  // the frozen reaching oracle (placeholder-then-patch).
  auto PlacePHIsFor = [&](const LiveRange &LR, LaneBitmask Lane) {
    for (const VNInfo *VNI : LR.valnos) {
      if (!VNI || VNI->isUnused() || !VNI->isPHIDef())
        continue;
      MachineBasicBlock *JoinMBB = LIS.getMBBFromIndex(VNI->def);
      if (!JoinMBB)
        continue;
      if (MachineOperand *PHIResult =
              createPHIInBlockReaching(*JoinMBB, OrigVReg, Lane))
        AllCreatedPHIs.push_back({PHIResult, Lane});
    }
  };
  if (FrozenOrigLI) {
    if (FrozenOrigLI->hasSubRanges()) {
      for (const LiveInterval::SubRange &S : FrozenOrigLI->subranges())
        if ((S.LaneMask & DefMask).any())
          PlacePHIsFor(S, S.LaneMask);
    } else {
      PlacePHIsFor(*FrozenOrigLI, DefMask);
    }
  }
  LLVM_DEBUG(dbgs() << "  PHI insertion complete. Created "
                    << AllCreatedPHIs.size() << " PHI registers total.\n");
  return AllCreatedPHIs;
}

// Reaching-VNI path (approach A): one PHI for a single subrange-aligned lane
// group. Every predecessor operand has exactly one reaching piece -> a single
// source (renamed reaching def) or an OrigVReg.subIdx placeholder patched
// later.
MachineOperand *MachineLaneSSAUpdater::createPHIInBlockReaching(
    MachineBasicBlock &JoinMBB, Register OrigVReg, LaneBitmask Lane) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const TargetRegisterClass *OrigRC = MRI.getRegClass(OrigVReg);
  const LaneBitmask FullMask = MRI.getMaxLaneMaskForVReg(OrigVReg);

  // Dedup: reuse a PHI already built for this (block, lane) this session.
  auto Key = std::make_pair(&JoinMBB, Lane);
  if (Register Existing = LanePHIs.lookup(Key))
    return &MRI.getVRegDef(Existing)->getOperand(0);

  // PHI result class = subregister class for Lane (full reg when Lane == full).
  const TargetRegisterClass *RC = OrigRC;
  unsigned LaneSub =
      (Lane == FullMask) ? 0 : getSubRegIndexForLaneMask(Lane, &TRI);
  if (LaneSub)
    if (const TargetRegisterClass *SubRC =
            TRI.getSubRegisterClass(OrigRC, LaneSub))
      RC = SubRC;

  Register PHIVReg = MRI.createVirtualRegister(RC);
  auto PHINode = BuildMI(JoinMBB, JoinMBB.begin(), DebugLoc(),
                         TII->get(TargetOpcode::PHI), PHIVReg);
  LLVM_DEBUG(dbgs() << "    createPHIInBlockReaching in BB#"
                    << JoinMBB.getNumber()
                    << " OrigVReg=" << printReg(OrigVReg, &TRI) << " Lane="
                    << PrintLaneMask(Lane) << " -> " << PHIVReg << "\n");

  for (MachineBasicBlock *Pred : JoinMBB.predecessors()) {
    SlotIndex EndP = LIS.getMBBEndIdx(Pred);
    VNInfo *V = reachingVNIForLaneGroup(*FrozenOrigLI, Lane, EndP);

    if (const RenamedDef *RD = renamedForReachingVNI(V)) {
      // Extract Lane from the renamed vreg: rebase Lane into its namespace.
      LaneBitmask RLane = rebaseLaneMask(Lane, RD->OrigLanes);
      LaneBitmask RFull = MRI.getMaxLaneMaskForVReg(RD->VReg);
      unsigned Sub =
          (RLane == RFull) ? 0 : getSubRegIndexForLaneMask(RLane, &TRI);
      PHINode.addReg(RD->VReg, 0, Sub);
    } else {
      // Placeholder: Root value (final) or a not-yet-materialized value patched
      // later by its owner's rewriteDominatedUses.
      unsigned Sub =
          (Lane == FullMask) ? 0 : getSubRegIndexForLaneMask(Lane, &TRI);
      PHINode.addReg(OrigVReg, 0, Sub);
    }
    PHINode.addMBB(Pred);
  }

  MachineInstr *PHI = PHINode.getInstr();
  LIS.InsertMachineInstrInMaps(*PHI);
  LanePHIs[Key] = PHIVReg;
  LLVM_DEBUG(dbgs() << "      Created: "; PHI->print(dbgs()));
  return &PHI->getOperand(0);
}

void MachineLaneSSAUpdater::rewriteUseReaching(
    Register OrigVReg, Register NewSSA, LaneBitmask MaskToRewrite,
    MachineInstr *DefMI, MachineInstr *UseMI, MachineOperand &MO,
    LaneBitmask OpMask, LiveInterval &OrigLI) {
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Point at which we query the reaching value: end of the predecessor for a
  // PHI operand, the use instruction otherwise.
  SlotIndex UsePt;
  if (UseMI->isPHI()) {
    unsigned OpIdx = UseMI->getOperandNo(&MO);
    MachineBasicBlock *Pred = UseMI->getOperand(OpIdx + 1).getMBB();
    UsePt = LIS.getMBBEndIdx(Pred);
  } else {
    UsePt = LIS.getInstructionIndex(*UseMI).getRegSlot();
  }

  // Lanes (within MaskToRewrite) whose reaching value at the use IS this def.
  // A real (non-PHI) def matches the frozen VNInfo at its slot; an inserted PHI
  // result matches the frozen PHI-def VNInfo at the PHI's block (the frozen LI
  // predates our PHIs and records merges as block-boundary PHI-defs, so PHI
  // results can't be matched by instruction slot).
  const bool DefIsPHI = DefMI->isPHI();
  MachineBasicBlock *DefBB = DefMI->getParent();
  // A real def's VNInfo sits at the early-clobber slot iff its def operand is
  // early-clobber (e.g. S_LOAD_..._ec); otherwise at the regular slot. Use the
  // exact slot so we match only THIS def, never another def on the same instr.
  SlotIndex DefSlot;
  if (!DefIsPHI) {
    bool EC = false;
    for (const MachineOperand &D : DefMI->defs())
      if (D.getReg() == NewSSA) {
        EC = D.isEarlyClobber();
        break;
      }
    DefSlot = LIS.getInstructionIndex(*DefMI).getRegSlot(EC);
  }
  SmallVector<std::pair<LaneBitmask, VNInfo *>, 4> Pieces;
  collectReachingVNIs(*FrozenOrigLI, OpMask & MaskToRewrite, UsePt, Pieces);
  LaneBitmask Owned = LaneBitmask::getNone();
  for (auto &[L, V] : Pieces) {
    if (!V)
      continue;
    bool Match = DefIsPHI
                     ? (V->isPHIDef() && LIS.getMBBFromIndex(V->def) == DefBB)
                     : (!V->isPHIDef() && V->def == DefSlot);
    if (Match)
      Owned |= L;
  }

  if (Owned.none())
    return; // this def provides none of the use's lanes; owner patches later

  LLVM_DEBUG(dbgs() << "    [reaching] use OpMask=" << PrintLaneMask(OpMask)
                    << " Owned=" << PrintLaneMask(Owned) << ": ";
             UseMI->print(dbgs()));

  if (Owned == OpMask) {
    // Whole operand provided by this def: use NewSSA with OpMask's subreg
    // (rebased into NewSSA's namespace; 0 when it is the full register).
    LaneBitmask NewLanes = rebaseLaneMask(OpMask, MaskToRewrite);
    LaneBitmask NewFull = MRI.getMaxLaneMaskForVReg(NewSSA);
    unsigned Sub =
        (NewLanes == NewFull) ? 0 : getSubRegIndexForLaneMask(NewLanes, &TRI);
    MO.setReg(NewSSA);
    MO.setSubReg(Sub);
    return;
  }

  // Partial: compose Owned lanes from NewSSA; the remaining lanes stay OrigVReg
  // (Root/final or a placeholder patched by their owner). The REG_SEQUENCE
  // result is a fresh whole register that only needs to hold OpMask-many lanes.
  // The operand may name an UNALIGNED subregister of OrigVReg (e.g.
  // sub1_sub2_sub3 of an sgpr_128, formed while iteratively rebuilding a
  // partial-def chain), for which getSubRegisterClass(OpRC, MO.getSubReg()) is
  // null. Derive the class from the base-0 (rebased) lane mask instead, which
  // maps to an aligned, valid subregister index of OpRC.
  const TargetRegisterClass *OpRC = MRI.getRegClass(MO.getReg());
  LaneBitmask FullRC = MRI.getMaxLaneMaskForVReg(MO.getReg());
  LaneBitmask RebasedUse = rebaseLaneMask(OpMask, OpMask);
  unsigned UseSub =
      (RebasedUse == FullRC) ? 0 : getSubRegIndexForLaneMask(RebasedUse, &TRI);
  const TargetRegisterClass *UseRC =
      UseSub ? TRI.getSubRegisterClass(OpRC, UseSub) : OpRC;
  assert(UseRC && "use operand subreg has no register class");
  SmallVector<LaneBitmask, 4> LanesToExtend;
  SlotIndex RSIdx;
  Register RSReg = buildRSForSuperUse(UseMI, MO, OrigVReg, NewSSA, Owned,
                                      OrigLI, UseRC, RSIdx, LanesToExtend);
  extendAt(OrigLI, RSIdx, LanesToExtend);
  MO.setReg(RSReg);
  MO.setSubReg(0);
  LIS.extendToIndices(LIS.getInterval(RSReg), {UsePt});
  updateDeadFlags(RSReg);
}

void MachineLaneSSAUpdater::rewriteDominatedUses(Register OrigVReg,
                                                 Register NewSSA,
                                                 LaneBitmask MaskToRewrite) {
  LLVM_DEBUG(dbgs() << "MachineLaneSSAUpdater::rewriteDominatedUses OrigVReg="
                    << OrigVReg << " NewSSA=" << NewSSA
                    << " Mask=" << PrintLaneMask(MaskToRewrite) << "\n");

  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Find the definition instruction for NewSSA
  MachineInstr *DefMI = MRI.getVRegDef(NewSSA);
  if (!DefMI) {
    LLVM_DEBUG(dbgs() << "  No definition found for NewSSA, skipping\n");
    return;
  }

  MachineBasicBlock *DefBB = DefMI->getParent();

  LLVM_DEBUG(dbgs() << "  Rewriting uses dominated by definition in BB#"
                    << DefBB->getNumber() << ": ");
  LLVM_DEBUG(DefMI->print(dbgs()));

  // Get OrigVReg's LiveInterval for reference
  LiveInterval &OrigLI = LIS.getInterval(OrigVReg);

  // Iterate through all uses of OrigVReg
  for (MachineOperand &MO :
       llvm::make_early_inc_range(MRI.use_operands(OrigVReg))) {
    MachineInstr *UseMI = MO.getParent();

    // Skip the definition instruction itself
    if (UseMI == DefMI)
      continue;

    // Get the lane mask for this operand
    LaneBitmask OpMask = operandLaneMask(MO);
    if ((OpMask & MaskToRewrite).none())
      continue;

    // Reaching-VNI ownership: this def owns exactly the OpMask lanes whose
    // reaching value at the use is this def's own VNInfo (replaces the old
    // block-dominance filter, which over-/under-claimed lanes -- Bucket 3).
    rewriteUseReaching(OrigVReg, NewSSA, MaskToRewrite, DefMI, UseMI, MO,
                       OpMask, OrigLI);
  }

  LLVM_DEBUG(dbgs() << "  Completed rewriting dominated uses\n");
}

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

/// Decompose \p Mask by OrigVReg's subranges and append {laneSubmask, reaching
/// VNInfo at \p Idx} per covered piece. Pieces partition \p Mask (subranges
/// partition all lanes), so the caller uses a single value when one piece
/// covers all of Mask, else composes a REG_SEQUENCE. A null VNInfo means dead
/// lanes. A reg without subranges yields one {Mask, mainVNI}.
void MachineLaneSSAUpdater::collectReachingVNIs(
    LiveInterval &OrigLI, LaneBitmask Mask, SlotIndex Idx,
    SmallVectorImpl<std::pair<LaneBitmask, VNInfo *>> &Out) {
  if (OrigLI.hasSubRanges()) {
    for (const LiveInterval::SubRange &S : OrigLI.subranges()) {
      LaneBitmask Piece = S.LaneMask & Mask;
      if (Piece.none())
        continue;
      Out.push_back({Piece, S.getVNInfoBefore(Idx)});
    }
    return;
  }
  Out.push_back({Mask, OrigLI.getVNInfoBefore(Idx)});
}

VNInfo *MachineLaneSSAUpdater::reachingVNIForLaneGroup(LiveInterval &OrigLI,
                                                       LaneBitmask Lane,
                                                       SlotIndex Idx) {
  SmallVector<std::pair<LaneBitmask, VNInfo *>, 2> Pieces;
  collectReachingVNIs(OrigLI, Lane, Idx, Pieces);
  assert(Pieces.size() <= 1 &&
         "reachingVNIForLaneGroup: lane group must lie within one subrange");
  return Pieces.empty() ? nullptr : Pieces.front().second;
}

/// Map a reaching VNInfo to the RenamedDef its def was renamed to this session.
/// Returns null when the value is a PHI-def merge (resolved via placeholder-
/// then-patch) or when the def has not (yet) been renamed, so the caller keeps
/// an OrigVReg placeholder that a later repair patches.
const MachineLaneSSAUpdater::RenamedDef *
MachineLaneSSAUpdater::renamedForReachingVNI(const VNInfo *V) {
  if (!V || V->isUnused() || V->isPHIDef())
    return nullptr;
  MachineInstr *DefMI = LIS.getInstructionFromIndex(V->def);
  if (!DefMI)
    return nullptr;
  auto It = DefInstrToRenamed.find(DefMI);
  return It != DefInstrToRenamed.end() ? &It->second : nullptr;
}

/// Check whether \p UseMI (a use of \p OrigVReg) is reachable from \p DefMI in
/// the CFG. Since OrigVReg is SSA, the value is live along every def->use path,
/// so plain CFG reachability exactly answers "is this use downstream of the
/// def" -- no dominance-frontier needed.
bool MachineLaneSSAUpdater::isUseReachableFromDef(MachineInstr *DefMI,
                                                  MachineInstr *UseMI,
                                                  Register OrigVReg) {
  MachineBasicBlock *DefBlock = DefMI->getParent();
  MachineOperand *UseOp =
      UseMI->findRegisterUseOperand(OrigVReg, &TRI, /*isKill=*/false);
  assert(UseOp && "UseMI must use OrigVReg");

  // Target of the query: the use's block, or -- for a PHI use -- the block the
  // value flows in from.
  const bool IsPHI = UseMI->isPHI();
  MachineBasicBlock *Target = UseMI->getParent();
  if (IsPHI) {
    unsigned OpIdx = UseMI->getOperandNo(UseOp);
    Target = UseMI->getOperand(OpIdx + 1).getMBB();
  }

  if (Target == DefBlock) {
    // A PHI incoming value leaves DefBlock after the def, so it is downstream.
    // A non-PHI same-block use is downstream iff the def precedes it; a use
    // *before* the def is reachable only via a back-edge (handled by the walk).
    if (IsPHI ||
        LIS.getInstructionIndex(*DefMI) < LIS.getInstructionIndex(*UseMI))
      return true;
  } else if (MDT.dominates(DefBlock, Target)) {
    // Fast path: dominance implies reachability.
    return true;
  }

  // General CFG reachability from DefBlock's successors. A block is its own
  // successor only through a back-edge, which correctly captures loop-carried
  // uses and excludes acyclic use-before-def.
  SmallPtrSet<const MachineBasicBlock *, 16> Visited;
  SmallVector<MachineBasicBlock *, 16> Worklist(DefBlock->succ_begin(),
                                                DefBlock->succ_end());
  while (!Worklist.empty()) {
    MachineBasicBlock *BB = Worklist.pop_back_val();
    if (BB == Target)
      return true;
    if (Visited.insert(BB).second)
      Worklist.append(BB->succ_begin(), BB->succ_end());
  }
  return false;
}

/// What lanes does this operand read?
LaneBitmask MachineLaneSSAUpdater::operandLaneMask(const MachineOperand &MO) {
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  if (unsigned Sub = MO.getSubReg())
    return TRI.getSubRegIndexLaneMask(Sub);
  return MRI.getMaxLaneMaskForVReg(MO.getReg());
}

/// Helper: Decompose a potentially non-contiguous lane mask into a vector of
/// subregister indices that together cover all lanes in the mask.
///
/// Key algorithm: Sort candidates by lane count (prefer larger subregs) to get
/// minimal covering set with largest possible subregisters.
///
/// Example: For vreg_128 with LaneMask = 0x0F | 0xF0 (sub0 + sub2, skipping
/// sub1)
///          Returns: [sub0_idx, sub2_idx] (not lo16, hi16, sub2, sub3)
static SmallVector<unsigned, 4>
getCoveringSubRegsForLaneMask(LaneBitmask Mask, const TargetRegisterInfo *TRI,
                              const TargetRegisterClass *RC) {
  if (Mask.none())
    return {};

  // Step 1: Collect all candidate subregisters that overlap with Mask
  SmallVector<unsigned, 4> Candidates;
  for (unsigned SubIdx = 1; SubIdx < TRI->getNumSubRegIndices(); ++SubIdx) {
    // Check if this subreg index is valid for this register class
    if (!TRI->getSubRegisterClass(RC, SubIdx))
      continue;

    LaneBitmask SubMask = TRI->getSubRegIndexLaneMask(SubIdx);
    // Add if it covers any lanes we need
    if ((SubMask & Mask).any()) {
      Candidates.push_back(SubIdx);
    }
  }

  // Step 2: Sort by number of lanes (descending) to prefer larger subregisters
  llvm::stable_sort(Candidates, [&](unsigned A, unsigned B) {
    return TRI->getSubRegIndexLaneMask(A).getNumLanes() >
           TRI->getSubRegIndexLaneMask(B).getNumLanes();
  });

  // Step 3: Greedily select subregisters, largest first
  SmallVector<unsigned, 4> OptimalSubIndices;
  for (unsigned SubIdx : Candidates) {
    LaneBitmask SubMask = TRI->getSubRegIndexLaneMask(SubIdx);
    // Only add if this subreg is fully contained in the remaining mask
    if ((Mask & SubMask) == SubMask) {
      OptimalSubIndices.push_back(SubIdx);
      Mask &= ~SubMask; // Remove covered lanes

      if (Mask.none())
        break; // All lanes covered
    }
  }

  return OptimalSubIndices;
}

/// Build a REG_SEQUENCE to materialize a super-reg/mixed-lane use.
/// Inserts at the PHI predecessor terminator (for PHI uses) or right before
/// UseMI otherwise. Returns the new full-width vreg, the RS index via OutIdx,
/// and the subrange lane masks that should be extended to that point.
Register MachineLaneSSAUpdater::buildRSForSuperUse(
    MachineInstr *UseMI, MachineOperand &MO, Register OldVR, Register NewVR,
    LaneBitmask MaskToRewrite, LiveInterval &LI,
    const TargetRegisterClass *UseRC, SlotIndex &OutIdx,
    SmallVectorImpl<LaneBitmask> &LanesToExtend) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  MachineBasicBlock *InsertBB = UseMI->getParent();
  MachineBasicBlock::iterator IP(UseMI);
  SlotIndex QueryIdx;

  if (UseMI->isPHI()) {
    unsigned OpIdx = UseMI->getOperandNo(&MO);
    MachineBasicBlock *Pred = UseMI->getOperand(OpIdx + 1).getMBB();
    InsertBB = Pred;
    IP = Pred->getFirstTerminator(); // ok if == end()
    QueryIdx = LIS.getMBBEndIdx(Pred).getPrevSlot();
  } else {
    QueryIdx = LIS.getInstructionIndex(*UseMI);
  }

  // Lanes the use needs. UseRC is already sized to exactly these lanes, so the
  // caller reads Dest whole.
  LaneBitmask UseMask = operandLaneMask(MO);

  // REG_SEQUENCE destination indices are in Dest's (narrower) namespace, while
  // the lane masks below are in OrigVReg's; re-base to convert.
  auto DestSub = [&](LaneBitmask Lanes) {
    return getSubRegIndexForLaneMask(rebaseLaneMask(Lanes, UseMask), &TRI);
  };

  Register Dest = MRI.createVirtualRegister(UseRC);
  auto RS = BuildMI(*InsertBB, IP,
                    (IP != InsertBB->end() ? IP->getDebugLoc() : DebugLoc()),
                    TII.get(TargetOpcode::REG_SEQUENCE), Dest);

  // Decompose into lanes from NewVR (updated) and lanes from OldVR (unchanged)
  LaneBitmask LanesFromNew = UseMask & MaskToRewrite;
  LaneBitmask LanesFromOld = UseMask & ~MaskToRewrite;

  LLVM_DEBUG(dbgs() << "        Building REG_SEQUENCE: UseMask="
                    << PrintLaneMask(UseMask)
                    << " LanesFromNew=" << PrintLaneMask(LanesFromNew)
                    << " LanesFromOld=" << PrintLaneMask(LanesFromOld) << "\n");

  SmallDenseSet<unsigned, 8> AddedSubIdxs;

  // Add source for lanes from NewVR (updated lanes)
  if (LanesFromNew.any()) {
    unsigned SubIdx = getSubRegIndexForLaneMask(LanesFromNew, &TRI);
    assert(SubIdx && "Failed to find subregister index for LanesFromNew");
    RS.addReg(NewVR, 0, 0).addImm(DestSub(LanesFromNew)); // NewVR whole
    AddedSubIdxs.insert(SubIdx);
    LanesToExtend.push_back(LanesFromNew);
  }

  // Add source for lanes from OldVR (unchanged lanes)
  // Handle both contiguous and non-contiguous lane masks
  // Non-contiguous example: Redefining only sub2 of vreg_128 leaves
  // LanesFromOld = sub0+sub1+sub3
  if (LanesFromOld.any()) {
    unsigned SubIdx = getSubRegIndexForLaneMask(LanesFromOld, &TRI);

    if (SubIdx) {
      // Contiguous case: single subregister covers all lanes
      RS.addReg(OldVR, 0, SubIdx).addImm(DestSub(LanesFromOld)); // OldVR.subIdx
      AddedSubIdxs.insert(SubIdx);
      LanesToExtend.push_back(LanesFromOld);
    } else {
      // Non-contiguous case: decompose into multiple subregisters
      const TargetRegisterClass *OldRC = MRI.getRegClass(OldVR);
      SmallVector<unsigned, 4> CoveringSubRegs =
          getCoveringSubRegsForLaneMask(LanesFromOld, &TRI, OldRC);

      assert(
          !CoveringSubRegs.empty() &&
          "Failed to decompose non-contiguous lane mask into covering subregs");

      LLVM_DEBUG(dbgs() << "        Non-contiguous LanesFromOld="
                        << PrintLaneMask(LanesFromOld) << " decomposed into "
                        << CoveringSubRegs.size() << " subregs\n");

      // Add each covering subregister as a source to the REG_SEQUENCE
      for (unsigned CoverSubIdx : CoveringSubRegs) {
        LaneBitmask CoverMask = TRI.getSubRegIndexLaneMask(CoverSubIdx);
        RS.addReg(OldVR, 0, CoverSubIdx)
            .addImm(DestSub(CoverMask)); // OldVR.CoverSubIdx
        AddedSubIdxs.insert(CoverSubIdx);
        LanesToExtend.push_back(CoverMask);

        LLVM_DEBUG(dbgs() << "          Added source: OldVR."
                          << TRI.getSubRegIndexName(CoverSubIdx) << " covering "
                          << PrintLaneMask(CoverMask) << "\n");
      }
    }
  }

  assert(!AddedSubIdxs.empty() && "REG_SEQUENCE must have at least one source");

  LIS.InsertMachineInstrInMaps(*RS);
  OutIdx = LIS.getInstructionIndex(*RS);

  // Create live interval for the REG_SEQUENCE result
  LIS.createAndComputeVirtRegInterval(Dest);

  // Extend live intervals of all source registers to cover this REG_SEQUENCE
  // Use the register slot to ensure the live range covers the use
  SlotIndex UseSlot = OutIdx.getRegSlot();
  for (MachineOperand &MO : RS.getInstr()->uses()) {
    if (MO.isReg() && MO.getReg().isVirtual()) {
      Register SrcReg = MO.getReg();
      LiveInterval &SrcLI = LIS.getInterval(SrcReg);
      LIS.extendToIndices(SrcLI, {UseSlot});
    }
  }

  LLVM_DEBUG(dbgs() << "        Built REG_SEQUENCE: ");
  LLVM_DEBUG(RS->print(dbgs()));

  return Dest;
}

/// Extend LI (and only the specified subranges) at Idx.
void MachineLaneSSAUpdater::extendAt(LiveInterval &LI, SlotIndex Idx,
                                     ArrayRef<LaneBitmask> Lanes) {
  SmallVector<SlotIndex, 1> P{Idx};
  LIS.extendToIndices(LI, P);
  for (auto &SR : LI.subranges())
    for (LaneBitmask L : Lanes)
      if (SR.LaneMask == L)
        LIS.extendToIndices(SR, P);
}

void MachineLaneSSAUpdater::updateDeadFlags(Register Reg) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  LiveInterval &LI = LIS.getInterval(Reg);
  MachineInstr *DefMI = MRI.getVRegDef(Reg);
  if (!DefMI)
    return;

  for (MachineOperand &MO : DefMI->defs()) {
    if (MO.getReg() == Reg && MO.isDead()) {
      // Check if this register is actually live (has uses)
      if (!LI.empty() && !MRI.use_nodbg_empty(Reg)) {
        MO.setIsDead(false);
        LLVM_DEBUG(dbgs() << "  Cleared dead flag on " << Reg << "\n");
      }
    }
  }
}
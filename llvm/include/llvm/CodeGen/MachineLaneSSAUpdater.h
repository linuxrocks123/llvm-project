//===- MachineLaneSSAUpdater.h - SSA repair for Machine IR (lane-aware) -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// === MachineLaneSSAUpdater Design Notes ===
//

#ifndef LLVM_CODEGEN_MACHINELANESSAUPDATER_H
#define LLVM_CODEGEN_MACHINELANESSAUPDATER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"        // SmallVector
#include "llvm/ADT/bit.h"                 // countr_zero
#include "llvm/CodeGen/LiveInterval.h"    // LiveRange
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/Register.h"       // Register
#include "llvm/CodeGen/SlotIndexes.h"    // SlotIndex
#include "llvm/CodeGen/TargetRegisterInfo.h" // For inline function
#include "llvm/MC/LaneBitmask.h"        // LaneBitmask
#include "llvm/Support/Allocator.h"      // BumpPtrAllocator (VNInfo::Allocator)
#include <memory>                         // std::unique_ptr

namespace llvm {

// Forward declarations to avoid heavy includes in the header.
class MachineFunction;
class MachineBasicBlock;
class MachineInstr;
class LiveIntervals;
class LiveRange;
class MachineDominatorTree;
class MachinePostDominatorTree; // optional if you choose to use it

//===----------------------------------------------------------------------===//
// MachineLaneSSAUpdater: universal SSA repair for Machine IR (lane-aware)
//
// Primary Use Case: repairSSAForNewDef()
//   - Caller creates a new instruction that defines an existing vreg (violating SSA)
//   - This function creates a new vreg (or uses a caller-provided one), 
//     replaces the operand, and repairs SSA
//   - Example: Insert "OrigVReg = ADD ..." and call repairSSAForNewDef()
//   - Works for full register and subregister definitions
//   - Handles all scenarios including spill/reload
//
// Advanced Usage: Caller-provided NewVReg
//   - By default, repairSSAForNewDef() creates a new virtual register automatically
//   - For special cases (e.g., subregister reloads where the spiller already 
//     created a register of a specific class), caller can provide NewVReg
//   - This gives full control over register class selection when needed
//===----------------------------------------------------------------------===//
class MachineLaneSSAUpdater {
public:
  MachineLaneSSAUpdater(MachineFunction &MF,
                        LiveIntervals &LIS,
                        MachineDominatorTree &MDT,
                        const TargetRegisterInfo &TRI)
      : MF(MF), LIS(LIS), MDT(MDT), TRI(TRI) {}

  // Repair SSA for a new definition that violates SSA form
  //
  // Parameters:
  //   NewDefMI: Instruction with a def operand that currently defines OrigVReg
  //   (violating SSA) 
  //   OrigVReg: The virtual register being redefined
  //   PHIRegDefOps: A vector of def operands for the PHI registers that were
  //   created
  // This function will:
  //   1. Find the def operand in NewDefMI that defines OrigVReg
  //   2. Derive the lane mask from the operand's subreg index (if any)
  //   3. Create a new virtual register with same class as OrigVReg
  //   4. Replace the operand in NewDefMI to define the new vreg (preserving
  //   subreg index)
  //   5. Perform SSA repair (insert PHIs, rewrite uses)
  //
  // Returns: The newly created SSA-repaired virtual register
  Register repairSSAForNewDef(MachineInstr &NewDefMI, Register OrigVReg,
                              SmallVectorImpl<MachineOperand *> &PHIRegDefOps);

  /// Invalidate the per-OrigVReg repair session so the next repairSSAForNewDef
  /// re-freezes OrigVReg's interval. Required when new defs (e.g. spiller
  /// reloads) are added incrementally between repair calls for the same
  /// OrigVReg; otherwise the cached FrozenOrigLI is stale and reaching
  /// resolution misses the new def.
  void resetSession() { RenameSessionOrig = Register(); }

  /// Check whether \p UseMI (a use of \p OrigVReg) is reachable from \p DefMI
  /// in the CFG -- i.e. whether the use lies downstream of the def (for a PHI
  /// use, the target is the incoming predecessor block). Used by the spiller to
  /// decide which uses a spill at \p DefMI affects. Since OrigVReg is SSA the
  /// value is live along every def->use path, so plain CFG reachability is
  /// exact -- no dominance-frontier needed.
  bool isUseReachableFromDef(MachineInstr *DefMI, MachineInstr *UseMI,
                             Register OrigVReg);

  /// Rewrite dominated uses of OrigVReg to NewSSA according to the
  /// exact/subset/super policy; create REG_SEQUENCE only when needed.
  void rewriteDominatedUses(Register OrigVReg,
                            Register NewSSA,
                            LaneBitmask MaskToRewrite);

  /// Reaching-VNI path: rewrite a single use operand \p MO. Owns exactly the
  /// \p OpMask lanes whose reaching value at the use is this def (\p DefMI /
  /// \p NewSSA covering \p MaskToRewrite). Owned==OpMask -> direct NewSSA subreg;
  /// partial -> REG_SEQUENCE (owned from NewSSA, rest OrigVReg placeholder);
  /// none -> no change.
  void rewriteUseReaching(Register OrigVReg, Register NewSSA,
                          LaneBitmask MaskToRewrite, MachineInstr *DefMI,
                          MachineInstr *UseMI, MachineOperand &MO,
                          LaneBitmask OpMask, LiveInterval &OrigLI);

private:
  // Common SSA repair logic
  // Returns a vector of MachineOperand pointers to the PHI result registers
  SmallVector<MachineOperand *> performSSARepair(Register NewVReg,
                                                     Register OrigVReg,
                                                     LaneBitmask DefMask,
                                                     MachineBasicBlock *DefBB);

  // Optional knobs (fluent style); no-ops until implemented in .cpp.
  MachineLaneSSAUpdater &setUndefEdgePolicy(bool MaterializeImplicitDef) {
    UndefEdgeAsImplicitDef = MaterializeImplicitDef; return *this; }
  MachineLaneSSAUpdater &setVerifyOnExit(bool Enable) {
    VerifyOnExit = Enable; return *this; }

  // --- Internal helpers ---

  // Index MI in SlotIndexes / LIS maps immediately after insertion.
  // Returns the SlotIndex assigned to the instruction.
  SlotIndex indexNewInstr(MachineInstr &MI);

  // Extend the main live range and the specific subranges at MI's index
  // for the lanes actually used/defined.
  void extendPreciselyAt(const Register VReg,
                         const SmallVector<LaneBitmask, 4> &LaneMasks,
                         const MachineInstr &AtMI);

  // Insert lane-aware Machine PHIs at the join points recorded as PHI-def
  // VNInfos in OrigVReg's frozen interval (per lane, the pruned IDF that
  // LiveIntervalCalc already computed). Returns each PHI result operand paired
  // with the OrigVReg lane it covers, so the caller can rewrite that PHI's uses
  // with the correct per-PHI lane (one PHI per subrange).
  SmallVector<std::pair<MachineOperand *, LaneBitmask>>
  insertLaneAwarePHI(Register OrigVReg, LaneBitmask DefMask);

  // Reaching-VNI path (approach A): create ONE PHI for a single lane group
  // (subrange-aligned \p Lane) at \p JoinMBB, resolving each predecessor operand
  // from the frozen OrigVReg reaching oracle (single source or OrigVReg
  // placeholder). Dedups via LanePHIs. Returns the PHI result operand.
  MachineOperand *createPHIInBlockReaching(MachineBasicBlock &JoinMBB,
                                           Register OrigVReg, LaneBitmask Lane);

  // Per-OrigVReg repair session key: reset when repairSSAForNewDef is first
  // called for a different OrigVReg (the driver processes all defs of an
  // OrigVReg before moving on).
  Register RenameSessionOrig;

  // A renamed def: the fresh SSA vreg plus the OrigVReg-namespace lanes it
  // covers. OrigLanes lets us rebase a target lane into VReg's own namespace to
  // extract the right subreg when VReg is wider than the queried lane (e.g. a
  // full def feeding a sub0 PHI operand).
  struct RenamedDef {
    Register VReg;
    LaneBitmask OrigLanes;
  };

  // Per-OrigVReg repair session: maps each renamed real (non-PHI) def
  // instruction to its RenamedDef. PHI-operand and use resolution map a reaching
  // VNInfo to its def instruction and look it up here. Only real defs are
  // stored; PHI-def reaching VNIs (block-boundary merges) are resolved via
  // placeholder-then-patch, so a MachineInstr* key suffices and is cheaper than
  // SlotIndex (no DenseMapInfo<SlotIndex>, O(1), no node allocs). Reset at the
  // start of each session.
  DenseMap<MachineInstr *, RenamedDef> DefInstrToRenamed;

  // Frozen deep copy of OrigVReg's LiveInterval, taken at session start before
  // any rename: the STABLE reaching-def oracle. Renames done during the session
  // strip renamed defs' VNInfos from the live LIS interval, so reaching queries
  // must run against this frozen copy instead. Rebuilt each session; VNInfos are
  // owned by FrozenAlloc (def SlotIndexes and isPHIDef are preserved by the
  // copy, which is all the lookup needs).
  BumpPtrAllocator FrozenAlloc;
  std::unique_ptr<LiveInterval> FrozenOrigLI;

  // Dedup for the reaching-VNI path's per-subrange PHIs (approach A): reuse the
  // PHI already built for a (join block, lane group) instead of creating a
  // duplicate when several defs share an IDF block. Reset per session.
  DenseMap<std::pair<MachineBasicBlock *, LaneBitmask>, Register> LanePHIs;

  // Reaching-def oracle (sound successor to dominance+order for lane decisions).
  // collectReachingVNIs: decompose \p Mask by OrigVReg's (old, still-valid)
  // LiveInterval subranges and append, for each covered piece, {laneSubmask,
  // reaching VNInfo at \p Idx}. The pieces partition \p Mask, so a caller can
  // use a single value when one piece covers all of Mask, or build a
  // REG_SEQUENCE across pieces otherwise. A null VNInfo means that piece's lanes
  // are dead at \p Idx. For a reg without subranges, returns one {Mask, mainVNI}.
  void collectReachingVNIs(
      LiveInterval &OrigLI, LaneBitmask Mask, SlotIndex Idx,
      SmallVectorImpl<std::pair<LaneBitmask, VNInfo *>> &Out);
  // Convenience wrapper for a single subrange-aligned lane group: returns the
  // one reaching VNInfo at \p Idx (null if the lane is dead there). Asserts if
  // \p Lane straddles subranges (which would yield more than one piece).
  VNInfo *reachingVNIForLaneGroup(LiveInterval &OrigLI, LaneBitmask Lane,
                                  SlotIndex Idx);
  // renamedForReachingVNI: map a reaching VNInfo to the RenamedDef its def was
  // renamed to this session, or null if it is a PHI-def merge or has not been
  // renamed (caller then leaves an OrigVReg placeholder to be patched later).
  const RenamedDef *renamedForReachingVNI(const VNInfo *V);

  // Internal helper methods for use rewriting
  LaneBitmask operandLaneMask(const MachineOperand &MO);
  Register buildRSForSuperUse(MachineInstr *UseMI, MachineOperand &MO,
                             Register OldVR, Register NewVR, LaneBitmask MaskToRewrite,
                             LiveInterval &LI, const TargetRegisterClass *UseRC,
                             SlotIndex &OutIdx, SmallVectorImpl<LaneBitmask> &LanesToExtend);
  void extendAt(LiveInterval &LI, SlotIndex Idx, ArrayRef<LaneBitmask> Lanes);
  void updateDeadFlags(Register Reg);

  // --- Data members ---
  MachineFunction &MF;
  LiveIntervals &LIS;
  MachineDominatorTree &MDT;
  const TargetRegisterInfo &TRI;

  bool UndefEdgeAsImplicitDef = true; // policy hook
  bool VerifyOnExit = true;           // run MF.verify()/LI.verify() at end
};

/// Get the subregister index that corresponds to the given lane mask.
/// \param Mask The lane mask to convert to a subregister index
/// \param TRI The target register info (provides target-specific subregister mapping)
/// \return The subregister index, or 0 if no single subregister matches
inline unsigned getSubRegIndexForLaneMask(LaneBitmask Mask, const TargetRegisterInfo *TRI) {
  if (Mask.none())
    return 0; // No subregister
  
  // Iterate through all subregister indices to find a match
  for (unsigned SubIdx = 1; SubIdx < TRI->getNumSubRegIndices(); ++SubIdx) {
    LaneBitmask SubMask = TRI->getSubRegIndexLaneMask(SubIdx);
    if (SubMask == Mask) {
      return SubIdx;
    }
  }
  
  // No exact match found - this might be a composite mask requiring REG_SEQUENCE
  return 0;
}

/// Re-express a lane mask from a super-register's namespace into the namespace
/// of a sub-register. \p Base is the sub-register's lane mask (in the super's
/// namespace); the result shifts \p Lanes down so Base's first lane sits at
/// bit 0. Identity when Base already starts at lane 0.
inline LaneBitmask rebaseLaneMask(LaneBitmask Lanes, LaneBitmask Base) {
  return LaneBitmask(Lanes.getAsInteger() >>
                     llvm::countr_zero(Base.getAsInteger()));
}

// DenseMapInfo specialization for LaneBitmask
template<>
struct DenseMapInfo<LaneBitmask> {
  static inline LaneBitmask getEmptyKey() {
    // Use a specific bit pattern for empty key
    return LaneBitmask(~0U - 1);
  }
  
  static inline LaneBitmask getTombstoneKey() {
    // Use a different bit pattern for tombstone  
    return LaneBitmask(~0U);
  }
  
  static unsigned getHashValue(const LaneBitmask &Val) {
    return (unsigned)Val.getAsInteger();
  }
  
  static bool isEqual(const LaneBitmask &LHS, const LaneBitmask &RHS) {
    return LHS == RHS;
  }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINELANESSAUPDATER_H
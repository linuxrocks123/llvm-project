//===--------------- AMDGPUSSARegisterSpiller.h  -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief SSA-aware Register Spiller for AMDGPU
///
/// This pass implements register spilling using the MachineLaneSSAUpdater
/// to maintain SSA form. Based on the approach from:
/// "Register Spilling and Live-Range Splitting for SSA-Form Programs"
/// Matthias Braun and Sebastian Hack, CC'09
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERSPILLER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERSPILLER_H

#include "AMDGPUNextUseAnalysis.h"
#include "GCNRegPressure.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "VRegMaskPair.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLaneSSAUpdater.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {

/// Helper data structure for grouping together uses where the head of the group
/// dominates all the other uses in the group. This allows us to emit a single
/// reload at the head, and all dominated uses in the group can reuse that
/// value.
class DomGroup {
  SmallVector<MachineInstr *> Uses;
  bool Deleted = false;

public:
  DomGroup(MachineInstr *MI) { Uses.push_back(MI); }
  MachineInstr *getHead() const { return Uses.front(); }
  bool isDeleted() const { return Deleted; }
  void merge(DomGroup &Other) {
    for (auto *MI : Other.Uses)
      Uses.push_back(MI);
    Other.Deleted = true;
  }
  const auto &getUses() const { return Uses; }
  size_t size() const { return Uses.size(); }
};

class AMDGPUSSARegisterSpiller : public MachineFunctionPass {
  const SIRegisterInfo *TRI = nullptr;
  const SIInstrInfo *TII = nullptr;
  const MachineLoopInfo *MLI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const SIMachineFunctionInfo *MFI = nullptr;
  MachineFrameInfo *FrameInfo = nullptr;
  LiveIntervals *LIS = nullptr;
  SlotIndexes *Indexes = nullptr;
  MachineDominatorTree *DT = nullptr;

  // SSA updater for IDF-based reachability and SSA repair
  // FIXME: Add cache invalidation when CFG changes
  std::unique_ptr<MachineLaneSSAUpdater> SSAUpdater;

  // Register pressure tracker (reused throughout the pass)
  std::unique_ptr<GCNUpwardRPTracker> RPTracker;

  // Next use analysis for spill candidate selection
  AMDGPUNextUseAnalysis::Result *NU = nullptr;

  // Stack slot management
  DenseMap<VRegMaskPair, int> Virt2StackSlotMap;

  // Track registers that have been stored at definition (to avoid EXEC drift)
  // When a register is selected for spilling, we store it right after definition
  // (when EXEC is full), then mark it dead at the "real spill" point using pruneValue()
  DenseMap<VRegMaskPair, MachineInstr *> StoredAtDefinition;

  // Divergent path handling: Maps spill instruction to reload instruction
  // for reachable but not dominated uses (divergent paths)
  DenseMap<MachineInstr *, SmallVector<MachineInstr *, 2>> SpillToReloadMap;

  // Register pressure limits (set during processFunction)
  unsigned VGPRLimit = 0;
  unsigned SGPRLimit = 0;

  // TODO: Add tracking for spilled/reloaded registers if needed for
  // verification

  /// Returns a stack slot for the given VRegMaskPair, creating one if needed.
  int assignVirt2StackSlot(VRegMaskPair VMP);

  /// Creates a spill slot for the given register class.
  int createSpillSlot(const TargetRegisterClass *RC);

  /// Calculates the total size of a register set in 32-bit register units.
  /// This accounts for AMDGPU's 32-bit physical register granularity:
  /// - VGPR_32: size 1
  /// - VReg_64: size 2
  /// - VReg_128: size 4, etc.
  unsigned getRegSetSizeInRegs(const VRegMaskPairSet &VRegs) const;

  /// Converts RPTracker's LiveRegSet to VRegMaskPairSet.
  VRegMaskPairSet
  convertLiveRegs(const GCNRPTracker::LiveRegSet &LiveRegs) const;

  /// Processes the entire function for one register class (SGPR or VGPR).
  /// This is called twice: first for SGPRs, then for VGPRs.
  bool processFunction(MachineFunction &MF, unsigned RPLimit, bool IsVGPRPass);

  /// Validates that final register pressure is within limits after all spilling.
  /// This is a temporary validation check until we properly handle clean path reloads.
  void validateFinalRegisterPressure(MachineFunction &MF, unsigned RPLimit,
                                      bool IsVGPR);

  /// Sorts the register set by next-use distance (descending).
  /// Registers with longer next-use distances are moved to the back.
  void sortRegSetByNextUse(MachineBasicBlock &MBB,
                           MachineBasicBlock::reverse_iterator I,
                           VRegMaskPairSet &Active);

  /// Spill selection algorithm: Selects which VRegMaskPairs to spill based on
  /// Belady's algorithm to reduce register pressure to the limit.
  ///
  /// This implements the core spill selection algorithm:
  /// 1. Calculate SizeToSpill = CurRP - RPLimit
  /// 2. Sort Active set by next-use distance (longest last)
  /// 3. Greedily select registers from the back until we've spilled enough
  /// 4. For registers larger than needed, split by subregister
  ///
  /// NOTE: This method only selects which registers to spill. The actual
  /// spill instruction emission is done by spillAndReload().
  ///
  /// Returns the set of VRegMaskPairs that were selected for spilling.
  VRegMaskPairSet getVMPsToSpill(MachineBasicBlock &MBB,
                                 MachineBasicBlock::reverse_iterator I,
                                 VRegMaskPairSet &Active, unsigned CurRP,
                                 unsigned RPLimit);

  /// High-level orchestration: Performs atomic spill+reload+SSA repair per
  /// register to keep MIR valid.
  ///
  /// IMPORTANT: Each register is completely spilled+reloaded+repaired before
  /// moving to the next to avoid invalid MIR state.
  ///
  /// Workflow:
  /// 1. Call getVMPsToSpill() to select and emit spill instructions
  /// 2. For each spilled VRegMaskPair, call emitReloadsAndRepairSSA()
  /// 3. MachineLaneSSAUpdater handles LiveInterval updates during SSA repair
  /// 4. Reset RPTracker after modifications
  ///
  /// Returns true if any spilling was performed.
  bool spillAndReload(MachineBasicBlock &MBB,
                      MachineBasicBlock::reverse_iterator I,
                      VRegMaskPairSet &Active, unsigned CurRP,
                      unsigned RPLimit);

  /// Stores register to stack slot right after its definition (when EXEC is full).
  /// This avoids EXEC drift issues by ensuring all lanes are stored before any
  /// divergent control flow can modify EXEC. Returns the store instruction.
  MachineInstr *spillAtDefinition(VRegMaskPair VMP);

  /// If the register was already stored at definition, uses pruneValue() to mark
  /// it dead at this point instead of emitting a new store.
  void spillBefore(MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator InsertBefore, VRegMaskPair VMP);

  /// Emits a spill instruction at the end of a basic block (before terminator).
  void spillAtEnd(MachineBasicBlock &MBB, VRegMaskPair VMP);

  /// Emits a spill instruction before the given position (reverse iterator).
  /// This is used during backward traversal in processFunction().
  void spillBefore(MachineBasicBlock &MBB,
                   MachineBasicBlock::reverse_iterator I, VRegMaskPair VMP);

  /// Emits a reload instruction before the given position (forward iterator).
  /// Does NOT perform SSA repair - only emits the instruction and registers it.
  /// Returns the reload instruction for later SSA repair.
  MachineInstr *emitReload(MachineBasicBlock::iterator InsertBefore,
                           VRegMaskPair VMP);

  /// Repairs SSA form for a reload instruction using MachineLaneSSAUpdater.
  /// Returns the new virtual register that holds the reloaded value.
  Register repairSSAForReload(MachineInstr *ReloadMI, VRegMaskPair VMP);

  /// Emits a reload instruction before the given position (forward iterator).
  /// Returns the new virtual register that holds the reloaded value.
  /// This is the primary reload method used during SSA repair.
  /// Convenience wrapper that calls emitReload() + repairSSAForReload().
  Register reloadBefore(MachineBasicBlock::iterator InsertBefore,
                        VRegMaskPair VMP);

  /// Emits a reload instruction at the end of a basic block (before
  /// terminator). Returns the new virtual register that holds the reloaded
  /// value.
  Register reloadAtEnd(MachineBasicBlock &MBB, VRegMaskPair VMP);

  /// Emits reload instructions for dominated uses and repairs SSA.
  ///
  /// Workflow:
  /// 1. Find all uses dominated by the spill instruction
  /// 2. Group uses by dominance chains (one reload per group)
  /// 3. Emit reload at each group head
  /// 4. Call MachineLaneSSAUpdater::repairSSAForNewDef() to:
  ///    - Handle non-dominated uses
  ///    - Insert PHIs where needed
  ///    - Rewrite all uses appropriately
  ///
  /// NOTE: We only optimize placement for dominated uses. MachineLaneSSAUpdater
  /// automatically handles all SSA repair including PHI insertion for reachable
  /// uses, so we don't need to manually compute IDF or classify uses.
  void emitReloadsAndRepairSSA(VRegMaskPair SpilledVMP, SlotIndex KillIdx,
                               int FrameIndex);

  /// Debug helper: dumps a register set to dbgs().
  void dumpRegSet(const VRegMaskPairSet &Regs) const;

  // ============================================================================
  // Divergent Path Optimization Helpers
  // ============================================================================
  
  /// Returns true if SpilledVMP has any uses on the path from StartBB to EndBB
  /// StopInstr: if provided, stop checking at this instruction in EndBB (exclusive)
  bool hasUseOnPath(MachineBasicBlock *StartBB, MachineBasicBlock *EndBB, 
                    VRegMaskPair SpilledVMP, MachineInstr *StopInstr = nullptr) const;
  
  /// Check if the given instruction still uses the spilled register with
  /// overlapping lane mask. Returns false if the use was rewritten by SSA repair.
  bool usesSpilledVMP(const MachineInstr *MI, VRegMaskPair SpilledVMP) const;
  
  /// Attempts to hoist spill to NCD if no uses exist on either path
  /// Returns true if hoisting was performed
  bool tryHoistSpillToNCD(MachineInstr *KillMI, VRegMaskPair SpilledVMP,
                          const SmallVectorImpl<MachineInstr *> &ReachableUses);
  
  /// Splits the join block at the reload point, placing reload on spill-path edge only.
  /// This ensures the reload executes only when arriving from the spill path, not the clean path.
  /// MachineLaneSSAUpdater will automatically insert PHI at the merge point.
  /// JoinBB is computed directly from CFG structure (no IDF needed).
  /// Returns the reload instruction in the split block (same ReloadMI pointer, but in new block).
  MachineInstr* splitBlockBeforeReload(MachineInstr *KillMI, 
                                       MachineInstr *ReloadMI, 
                                       VRegMaskPair SpilledVMP);
  
  /// Handles reachable but not dominated uses via split-before-use.
  /// With "store at definition", spills are correct (all lanes stored), so we only
  /// need to handle reload placement. No WWM needed since we store the same mask as defined.
  void handleReachableUse(MachineInstr *KillMI, MachineInstr *ReloadMI,
                          VRegMaskPair SpilledVMP);

public:
  static char ID;

  AMDGPUSSARegisterSpiller() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU SSA Register Spiller";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<SlotIndexesWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<AMDGPUNextUseAnalysisWrapper>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUSSAREGISTERSPILLER_H

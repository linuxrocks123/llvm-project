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

/// Dom-group: head instruction dominates a list of other uses.
/// Built once at spill decision time, reused during processing.
class DomGroup {
  MachineInstr *Head;
  SmallVector<MachineInstr *, 4> DominatedUses;

public:
  DomGroup(MachineInstr *MI) : Head(MI) {}
  
  MachineInstr *getHead() const { return Head; }
  const SmallVector<MachineInstr *, 4> &getDominatedUses() const { return DominatedUses; }
  
  void addDominatedUse(MachineInstr *MI) { DominatedUses.push_back(MI); }
  
  void promoteHead(MachineInstr *NewHead) {
    DominatedUses.push_back(Head);
    Head = NewHead;
  }
  
  size_t size() const { return 1 + DominatedUses.size(); }
};

/// SpillInfo: captures spill decision with pre-built dom-groups.
struct SpillInfo {
  VRegMaskPair SpilledVMP;
  SlotIndex KillIdx;
  int FrameIndex;
  SmallVector<DomGroup, 4> DomGroups;
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

  VRegMaskPairSet ReloadedRegs;

  // Register pressure limits (set during processFunction)
  unsigned VGPRLimit = 0;
  unsigned SGPRLimit = 0;

  // Current pass type for reload optimizer RP calculation
  bool IsVGPRPass = false;

  // Set when a reload redef is emitted (Option 3), i.e. SSA was broken and must
  // be reconstructed by the second RebuildSSA pass.
  bool SSAInvalidated = false;
  
  // Reload optimizer: cached max RP per block (cleared per spill analysis)
  DenseMap<MachineBasicBlock *, unsigned> MaxRPCache;

  // Track reloads per block to avoid duplicates
  DenseMap<std::pair<MachineBasicBlock *, Register>, Register> BlockReloadCache;

  // TODO: Add tracking for spilled/reloaded registers if needed for
  // verification

  /// Inserts a virtual spill marker at the given position.
  void insertVirtualSpillMarker(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I,
                                VRegMaskPair VMP);

  /// Returns a stack slot for the given VRegMaskPair, creating one if
  /// needed.
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
  /// Uses IsVGPRPass class member (set before calling).
  bool processFunction(MachineFunction &MF, unsigned RPLimit);

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

  /// Build dom-groups for a register at spill decision time.
  void buildDomGroupsForSpill(SpillInfo &Info);

  /// Emits reloads and repairs SSA using IDF-first PHI insertion.
  /// Uses pre-built dom-groups from SpillInfo.
  void emitReloadsAndRepairSSA(SpillInfo &Info);

  /// Process one PIDF block - insert PHI or reloads as needed.
  /// Returns the inserted PHI instruction, or nullptr if no PHI was needed.
  MachineInstr *processPIdfBlock(MachineBasicBlock *PIdfBB, VRegMaskPair SpilledVMP,
                                 MachineBasicBlock *KillBB,
                                 const SmallVectorImpl<DomGroup *> &Groups,
                                 unsigned RPLimit);

  /// Process kill-dominated groups using all DomGroups from Info.
  void processKillDominatedGroups(SpillInfo &Info, MachineBasicBlock *KillBB,
                                  unsigned RPLimit);

  /// Process a list of kill-dominated groups.
  void processKillDominatedGroupsWithList(const SmallVectorImpl<DomGroup *> &Groups,
                                          VRegMaskPair SpilledVMP,
                                          MachineBasicBlock *KillBB,
                                          MachineInstr *KillMI,
                                          unsigned RPLimit);

  /// Finalize live intervals after all reloads and use rewriting.
  void finalizeLiveIntervals(Register SpilledReg);

  /// Get or create reload in a block.
  /// If InsertBefore is provided, inserts before that instruction.
  /// Otherwise inserts at block end (before terminator) and caches the result.
  /// Returns {ReloadReg, ReloadMI}. ReloadMI is nullptr if using cached reload.
  std::pair<Register, MachineInstr *>
  getOrCreateReloadInBlock(MachineBasicBlock *BB, VRegMaskPair SpilledVMP,
                           MachineInstr *InsertBefore = nullptr);

  /// Insert reload for a use instruction. For PHI uses, inserts in predecessor blocks.
  /// For non-PHI uses, handles loop adjustment.
  bool insertReloadForUse(MachineInstr *UseMI, VRegMaskPair SpilledVMP,
                          MachineBasicBlock *KillBB);

  /// Emit reload to a specific register.
  MachineInstr *emitReloadToReg(MachineBasicBlock::iterator InsertBefore,
                                 VRegMaskPair VMP, Register TargetReg);

  /// Sort PIDF blocks by dominance order.
  void sortByDominanceOrder(SmallVectorImpl<MachineBasicBlock *> &Blocks);

  /// Find closest dominating PIDF block.
  MachineBasicBlock *findClosestDominatingPIDF(
      MachineInstr *UseMI,
      const SmallVectorImpl<MachineBasicBlock *> &PIdfBlocks);

  /// Debug helper: dumps a register set to dbgs().
  void dumpRegSet(const VRegMaskPairSet &Regs) const;

  // ============================================================================
  // Divergent Path Optimization Helpers
  // ============================================================================
  
  /// Walk BFS from \p StartBB through blocks where \p SpilledReg is live.
  /// For each block, finds first use of SpilledReg (if any) and calls IsBad.
  /// \param stopOnBad If true (default), return immediately when IsBad returns true.
  ///                  If false, continue walking all paths.
  /// \returns true if all paths OK, false if any bad found.
  bool walkPathsToUses(MachineBasicBlock *StartBB,
                       Register SpilledReg,
                       llvm::function_ref<bool(MachineBasicBlock *,
                                               MachineInstr *)> IsBad,
                       bool stopOnBad = true) const;
  
  /// Checks if the given block has any use of SpilledVMP.
  /// If StopInstr is provided and is in this block, only checks up to that instruction.
  /// Uses NextUseAnalysis for fast full-block checks, falls back to instruction scan
  /// for partial blocks.
  bool blockHasUse(MachineBasicBlock *BB, VRegMaskPair SpilledVMP,
                   MachineInstr *StopInstr) const;
  
  /// Check if the given instruction still uses the spilled register with
  /// overlapping lane mask. Returns false if the use was rewritten by SSA repair.
  bool usesSpilledVMP(const MachineInstr *MI, VRegMaskPair SpilledVMP) const;
  
  /// Attempts to hoist spill to NCD if no unexpected uses exist on paths.
  /// Returns NCD if hoisting succeeded (and moves virtual spill marker), nullptr otherwise.
  MachineBasicBlock *tryHoistSpillToNCD(MachineInstr *KillMI, VRegMaskPair SpilledVMP,
                                         const SmallVectorImpl<MachineInstr *> &ReachableUses);

  /// Collects all dominated blocks of the given spill block.
  void collectDominatedBlocks(MachineBasicBlock &SpillMBB,
                              SmallVectorImpl<MachineBasicBlock *> &DomBBs) const;

  void cutFromLiveRange(LiveRange &LR, SlotIndex CutStart, SlotIndex CutEnd);

  // ============================================================================
  // Reload Optimizer
  // ============================================================================
  
  /// Computes and caches maximum register pressure within a basic block.
  unsigned getMaxRPForBlock(MachineBasicBlock *MBB);
  
  /// Computes max RP in block walking down from StopMI to block start.
  unsigned getMaxRPInBlockDownTo(MachineBasicBlock *MBB, MachineInstr *StopMI);
  
  /// Checks if reload can be hoisted to NCD by walking paths and checking RP.
  /// InsertPoint is the reload insertion point in NCD (nullptr if no use in NCD).
  bool canHoistReloadTo(MachineBasicBlock *NCD, MachineInstr *InsertPoint,
                        unsigned RPLimit, Register SpilledReg);
  
  /// Optimize reload placement for multiple dom-group heads.
  /// Uses iterative greedy clique-based NCD algorithm with RP checking.
  /// Returns list of (ReloadBB, InsertBeforeMI) pairs.
  SmallVector<std::pair<MachineBasicBlock *, MachineInstr *>, 4>
  optimizeReloadPlacing(const SmallVectorImpl<MachineInstr *> &GroupHeads,
                        unsigned RPLimit, Register SpilledReg);
  
  /// Fix pathological PHIs that still use the spilled register.
  /// In triangle/diamond CFGs, a PHI may merge a reloaded value with the
  /// original spilled value. Replace such PHIs with reloads.
  void fixPathologicalPHIs(VRegMaskPair SpilledVMP, int FrameIndex,
                           MachineInstr *KillMI);

  // ============================================================================
  // Loop-Aware Spilling Helpers
  // ============================================================================
  
  /// Returns true if VMP's definition is inside any loop.
  bool hasDefInLoop(VRegMaskPair VMP) const;
  
  /// Returns true if VMP has any use inside a loop.
  bool hasUseInLoop(VRegMaskPair VMP) const;
  
  /// Find the loop exit block that dominates SpillBB.
  /// For loops with multiple exits, returns the one dominating SpillBB.
  /// Returns nullptr if no suitable exit exists.
  MachineBasicBlock *getLoopExitDominatingSpill(MachineLoop *Loop,
                                                 MachineBasicBlock *SpillBB) const;
  
  /// Get the effective kill block after hoisting out of all enclosing loops.
  /// If SpillBB is inside loop(s), returns the outermost loop's preheader.
  /// Otherwise returns SpillBB unchanged.
  MachineBasicBlock *getEffectiveKillBB(MachineBasicBlock *SpillBB) const;
  
  /// Adjust reload placement for loop-aware spilling.
  /// If ReloadBB is in a loop but KillBB is outside, returns the preheader
  /// (only if it doesn't cause RP to exceed limit on the path).
  /// Returns the adjusted (ReloadBB, InsertBeforeMI) pair.
  std::pair<MachineBasicBlock *, MachineInstr *>
  adjustReloadForLoop(MachineBasicBlock *ReloadBB, MachineInstr *InsertBeforeMI,
                      MachineBasicBlock *KillBB, Register SpilledReg);

  /// Accounting only: compute how many VGPRs SGPR-spill-to-lane will consume,
  /// so Pass 2 (VGPR) sees the correct budget. Does NOT lower the pseudos and
  /// does NOT reserve physregs — physical lane reservation and the actual
  /// writelane/readlane materialization happen later, at SGPR coloring time
  /// (SuperReg must be physical for SGPRSpillBuilder).
  /// Called between Pass 1 (SGPR) and Pass 2 (VGPR).
  unsigned countSGPRSpillVGPRs(MachineFunction &MF);

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

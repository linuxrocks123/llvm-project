//===--------------- AMDGPUSSARegisterSpiller.cpp  -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSSARegisterSpiller.h"
#include "AMDGPU.h"
#include "AMDGPUSSARAUtils.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/InitializePasses.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-spiller"

STATISTIC(NumSpills, "Number of register spills");
STATISTIC(NumReloads, "Number of register reloads");

// ============================================================================
// Helper function to identify spill instructions
// ============================================================================

static bool isSpillInstr(const MachineInstr *MI) {
  if (MI->getOpcode() == AMDGPU::SI_SPILL_S32_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S64_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S96_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S128_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S160_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S192_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S224_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S256_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S288_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S320_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S352_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S384_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S512_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S1024_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V32_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V64_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V96_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V128_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V160_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V192_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V224_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V256_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V288_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V320_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V352_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V384_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V512_SAVE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V1024_SAVE)
    return true;
  return false;
}

char AMDGPUSSARegisterSpiller::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUSSARegisterSpiller, DEBUG_TYPE,
                      "AMDGPU SSA Register Spiller", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AMDGPUNextUseAnalysisWrapper)
INITIALIZE_PASS_END(AMDGPUSSARegisterSpiller, DEBUG_TYPE,
                    "AMDGPU SSA Register Spiller", false, false)

int AMDGPUSSARegisterSpiller::assignVirt2StackSlot(VRegMaskPair VMP) {
  assert(VMP.getVReg().isVirtual() && "Expected virtual register");

  // Check if we already have a stack slot for this VRegMaskPair
  auto It = Virt2StackSlotMap.find(VMP);
  if (It != Virt2StackSlotMap.end())
    return It->second;

  // Create a new stack slot
  const TargetRegisterClass *RC = VMP.getRegClass(MRI, TRI);
  int FI = createSpillSlot(RC);
  Virt2StackSlotMap[VMP] = FI;
  return FI;
}

int AMDGPUSSARegisterSpiller::createSpillSlot(const TargetRegisterClass *RC) {
  unsigned SpillSize = TRI->getSpillSize(*RC);
  Align SpillAlign = TRI->getSpillAlign(*RC);
  return FrameInfo->CreateSpillStackObject(SpillSize, SpillAlign);
}

unsigned AMDGPUSSARegisterSpiller::getRegSetSizeInRegs(
    const VRegMaskPairSet &VRegs) const {
  unsigned Size = 0;
  for (const auto &VMP : VRegs) {
    // Use the 32-bit granularity size calculation from VRegMaskPair
    Size += VMP.getSizeInRegs(TRI);
  }
  return Size;
}

VRegMaskPairSet AMDGPUSSARegisterSpiller::convertLiveRegs(
    const GCNRPTracker::LiveRegSet &LiveRegs) const {
  VRegMaskPairSet Result;
  for (const auto &[Reg, Mask] : LiveRegs) {
    if (Register::isVirtualRegister(Reg)) {
      Result.insert(VRegMaskPair(Register(Reg), Mask));
    }
  }
  return Result;
}

void AMDGPUSSARegisterSpiller::processFunction(MachineFunction &MF,
                                               unsigned RPLimit,
                                               bool IsVGPRPass) {
  LLVM_DEBUG(dbgs() << "processFunction: " << (IsVGPRPass ? "VGPR" : "SGPR")
                    << " pass, limit=" << RPLimit << "\n");

  // Traverse basic blocks in reverse post-order (RPO)
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);

  for (MachineBasicBlock *MBB : RPOT) {
    LLVM_DEBUG(dbgs() << "\nProcessing " << printMBBReference(*MBB) << "\n");

    // Initialize GCNUpwardRPTracker for backward traversal
    GCNUpwardRPTracker RPTracker(*LIS);
    RPTracker.reset(*MBB);  // Reset to the end of the block

    // Traverse instructions backward (from end to beginning)
    // This is the correct order for register allocation and spilling
    for (auto I = MBB->rbegin(), E = MBB->rend(); I != E; ++I) {
      MachineInstr &MI = *I;

      LLVM_DEBUG(dbgs() << "  Processing: " << MI);

      // Move tracker to the state before this instruction
      RPTracker.recede(MI);

      // Get current register pressure
      GCNRegPressure CurPressure = RPTracker.getPressure();
      
      // Get pressure for the current pass using the appropriate API
      // Note: hasGFX90AInsts() determines if we have unified VGPR file
      const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
      unsigned CurRP = IsVGPRPass ? CurPressure.getVGPRNum(ST.hasGFX90AInsts())
                                  : CurPressure.getSGPRNum();

      LLVM_DEBUG(dbgs() << "    " << (IsVGPRPass ? "VGPR" : "SGPR")
                        << " pressure: " << CurRP << "\n");

      // Check if we need to spill
      if (CurRP >= RPLimit) {
        LLVM_DEBUG(dbgs() << "  " << (IsVGPRPass ? "VGPR" : "SGPR")
                          << " pressure " << CurRP << " >= limit " << RPLimit
                          << ", need to spill\n");

        // Determine the register kind for filtering
        GCNRegPressure::RegKind Kind = IsVGPRPass ? GCNRegPressure::VGPR 
                                                  : GCNRegPressure::SGPR;

        // Get the slot index for the current instruction
        SlotIndex Slot = LIS->getInstructionIndex(MI).getRegSlot();
        
        // Get live registers filtered by register kind using getLiveRegs helper
        GCNRPTracker::LiveRegSet LiveRegsMap = 
            llvm::getLiveRegs(Slot, *LIS, *MRI, Kind);
        
        // Convert to VRegMaskPairSet for spill candidate selection
        VRegMaskPairSet ActiveRegs = convertLiveRegs(LiveRegsMap);

        // Call spillAndReload to handle atomic spill+reload+SSA repair
        spillAndReload(*MBB, I, ActiveRegs, CurRP, RPLimit);
      }
    }
  }
}

void AMDGPUSSARegisterSpiller::sortRegSetByNextUse(
    MachineBasicBlock &MBB, MachineBasicBlock::reverse_iterator I,
    VRegMaskPairSet &Active) {
  // Pre-compute next-use distances for all registers to avoid redundant calls
  // during sorting (sort makes O(n log n) comparisons, but we only need O(n) distance calculations)
  DenseMap<VRegMaskPair, unsigned> DistanceMap;
  
  // Get the current instruction
  MachineInstr *MI = &(*I);
  
  // Calculate distance once for each register
  for (const auto &VMP : Active) {
    unsigned Dist = NU->getNextUseDistance(MI, VMP);
    DistanceMap[VMP] = Dist;
  }
  
  // Sort using pre-computed distances
  Active.sort([&](const VRegMaskPair &A, const VRegMaskPair &B) {
    // Shorter distance first (longer distance at back for spilling)
    return DistanceMap[A] < DistanceMap[B];
  });
  
  LLVM_DEBUG({
    dbgs() << "sortRegSetByNextUse: Active set sorted at " << *MI;
    for (const auto &VMP : Active) {
      dbgs() << "  " << printReg(VMP.getVReg(), TRI) 
             << " (mask " << PrintLaneMask(VMP.getLaneMask()) 
             << ") : " << DistanceMap[VMP] << "\n";
    }
  });
}

VRegMaskPairSet AMDGPUSSARegisterSpiller::getVMPsToSpill(
    MachineBasicBlock &MBB, MachineBasicBlock::reverse_iterator I,
    VRegMaskPairSet &Active, unsigned CurRP, unsigned RPLimit) {
  
  VRegMaskPairSet ToSpill;
  
  LLVM_DEBUG(dbgs() << "getVMPsToSpill(): CurRP=" << CurRP 
                    << ", RPLimit=" << RPLimit << "\n");

  // Step 1: Calculate how much we need to spill
  if (CurRP <= RPLimit) {
    LLVM_DEBUG(dbgs() << "getVMPsToSpill(): No spilling needed (RP <= limit)\n");
    return ToSpill;
  }

  unsigned SizeToSpill = CurRP - RPLimit;
  LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Need to spill " << SizeToSpill
                    << " 32-bit register units\n");

  // Step 2: Sort Active set by next-use distance (longest last)
  sortRegSetByNextUse(MBB, I, Active);

  // Step 3: Greedily select registers to spill from the back
  unsigned RemainingToSpill = SizeToSpill;

  LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Need to reduce RP by " 
                    << RemainingToSpill << " units\n");

  while (RemainingToSpill > 0 && !Active.empty()) {
    VRegMaskPair Candidate = Active.pop_back_val();
    unsigned CandidateSize = Candidate.getSizeInRegs(TRI);

    LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Considering candidate "
                      << printReg(Candidate.getVReg(), TRI) << " with mask "
                      << PrintLaneMask(Candidate.getLaneMask()) << " (size "
                      << CandidateSize << ")\n");

    // If this register is larger than what we need to spill, split it by
    // subregisters and only spill what's needed
    if (CandidateSize > RemainingToSpill) {
      LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Candidate is too large (" 
                        << CandidateSize << " > " << RemainingToSpill
                        << "), splitting by subregisters\n");

      // Get subregisters sorted by next-use distance (longest first)
      MachineInstr *MI = &(*I);
      SmallVector<VRegMaskPair> SortedSubregs = 
          NU->getSortedSubregUses(MI, Candidate);

      if (!SortedSubregs.empty()) {
        // Split by subregisters and spill only what's needed
        for (const auto &SubReg : SortedSubregs) {
          unsigned SubRegSize = SubReg.getSizeInRegs(TRI);
          
          ToSpill.insert(SubReg);
          RemainingToSpill -= SubRegSize;
          
          if (RemainingToSpill == 0)
            break;
        }
        
        // Note: We don't need to insert remaining lanes back into Active.
        // RPTracker will provide the correct live set on the next instruction.
      } else {
        // Fallback: no subregister info available, spill the whole register
        LLVM_DEBUG(dbgs() << "getVMPsToSpill(): No subregister info, "
                             "spilling whole register\n");
        ToSpill.insert(Candidate);
        RemainingToSpill = 0;
      }
    } else {
      // This register fits within our spill budget
      ToSpill.insert(Candidate);
      RemainingToSpill -= CandidateSize;
    }
  }

  LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Selected " << ToSpill.size()
                    << " VMP(s) for spilling:\n";
             dumpRegSet(ToSpill));

  return ToSpill;
}

bool AMDGPUSSARegisterSpiller::spillAndReload(
    MachineBasicBlock &MBB, MachineBasicBlock::reverse_iterator I,
    VRegMaskPairSet &Active, unsigned CurRP, unsigned RPLimit) {
  
  LLVM_DEBUG(dbgs() << "\n=== spillAndReload() at " << *I << "\n");
  LLVM_DEBUG(dbgs() << "CurRP=" << CurRP << ", RPLimit=" << RPLimit << "\n");

  // Step 1: Select which VRegMaskPairs to spill using Belady's algorithm
  VRegMaskPairSet ToSpill = getVMPsToSpill(MBB, I, Active, CurRP, RPLimit);
  
  if (ToSpill.empty()) {
    LLVM_DEBUG(dbgs() << "spillAndReload(): Nothing to spill\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "spillAndReload(): Will spill " << ToSpill.size() 
                    << " VMP(s)\n");

  // Step 2: For each selected VMP, perform atomic spill+reload+SSA repair
  // IMPORTANT: Process one VMP at a time to keep MIR valid
  for (const auto &VMP : ToSpill) {
    Register VReg = VMP.getVReg();
    LaneBitmask Mask = VMP.getLaneMask();
    
    LLVM_DEBUG(dbgs() << "\nspillAndReload(): Processing VMP " 
                      << printReg(VReg, TRI) 
                      << " with mask " << PrintLaneMask(Mask) << "\n");

    // Step 2a: Emit spill instruction
    spillBefore(MBB, I, VMP);
    
    // Get the spill instruction we just emitted
    MachineBasicBlock::iterator SpillPos = I.getReverse();
    MachineInstr *SpillMI = &*std::prev(SpillPos);
    
    LLVM_DEBUG(dbgs() << "spillAndReload(): Emitted spill: " << *SpillMI);
    
    // Step 2b: Get stack slot for reload phase
    int FI = assignVirt2StackSlot(VMP);
    
    // Step 2c: Emit reloads for dominated uses and repair SSA
    // TODO: Implement emitReloadsAndRepairSSA()
    // This should:
    // 1. Find all uses dominated by SpillMI
    // 2. Group dominated uses by dominance chains (DomGroup)
    // 3. Emit one reload per group at the group head
    // 4. Call MachineLaneSSAUpdater::repairSSAForNewDef() which will:
    //    - Handle all non-dominated uses
    //    - Insert PHIs where needed (using IDF internally)
    //    - Rewrite all uses appropriately
    
    // Step 2c: Emit reloads for dominated uses and repair SSA
    emitReloadsAndRepairSSA(VMP, SpillMI, FI);
  }

  LLVM_DEBUG(dbgs() << "spillAndReload(): Completed, spilled " 
                    << ToSpill.size() << " VMP(s)\n");
  LLVM_DEBUG(dbgs() << "===================================\n\n");

  return true;
}

void AMDGPUSSARegisterSpiller::emitReloadsAndRepairSSA(
    VRegMaskPair SpilledVMP, MachineInstr *SpillMI, int FrameIndex) {
  
  Register SpilledReg = SpilledVMP.getVReg();
  LaneBitmask SpilledMask = SpilledVMP.getLaneMask();
  
  LLVM_DEBUG(dbgs() << "\n=== emitReloadsAndRepairSSA() ===\n");
  LLVM_DEBUG(dbgs() << "Spilled: " << printReg(SpilledReg, TRI) 
                    << " with mask " << PrintLaneMask(SpilledMask) << "\n");
  LLVM_DEBUG(dbgs() << "SpillMI: " << *SpillMI);

  // Step 1: Find all uses dominated by the spill instruction
  SmallVector<MachineInstr *> DominatedUses;
  
  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(SpilledReg)) {
    // Skip the spill instruction itself
    if (&UseMI == SpillMI)
      continue;
    
    // Skip other spill instructions (they're not "real" uses we want to reload for)
    // This primarily handles previous spills of overlapping subregisters.
    // Note: Reload instructions will never appear here - they *define* the register,
    // they don't *use* it, so they won't be in the use list.
    if (isSpillInstr(&UseMI))
      continue;
    
    // Check if this use is dominated by the spill
    if (DT->dominates(SpillMI, &UseMI)) {
      DominatedUses.push_back(&UseMI);
      LLVM_DEBUG(dbgs() << "  Dominated use: " << UseMI);
    }
  }
  
  LLVM_DEBUG(dbgs() << "Found " << DominatedUses.size() 
                    << " dominated use(s)\n");
  
  // Step 2: Group uses by dominance chains
  // Each use starts in its own group. Then we merge groups where one head
  // dominates another, creating maximal dominance chains. This minimizes
  // the number of reload instructions.
  SmallVector<DomGroup, 2> Groups;
  
  // Initially, each use is its own group head
  for (auto *Use : DominatedUses) {
    Groups.emplace_back(Use);
  }
  
  // Merge groups: if head(G1) dominates head(G2), merge G2 into G1
  // This creates longest possible dominance chains
  for (unsigned Idx1 = 0, E = Groups.size(); Idx1 != E; ++Idx1) {
    auto &G1 = Groups[Idx1];
    if (G1.isDeleted())
      continue;
      
    for (unsigned Idx2 = Idx1 + 1; Idx2 < E; ++Idx2) {
      auto &G2 = Groups[Idx2];
      if (G2.isDeleted())
        continue;
        
      // If G1's head dominates G2's head, merge G2 into G1
      if (DT->dominates(G1.getHead(), G2.getHead())) {
        LLVM_DEBUG(dbgs() << "  Merging group: " << *G1.getHead() 
                          << "    dominates: " << *G2.getHead());
        G1.merge(G2);
      }
    }
  }
  
  // Count and report groups
  unsigned NumGroups = 0;
  for (const auto &G : Groups) {
    if (!G.isDeleted())
      ++NumGroups;
  }
  LLVM_DEBUG(dbgs() << "Created " << NumGroups << " dominance group(s)\n");
  
  // Step 3 & 4: Emit reload at each group head and repair SSA
  // For each group, we emit ONE reload at the head. The MachineLaneSSAUpdater
  // inside reloadBefore() will automatically:
  // - Create a new VReg
  // - Rewrite all dominated uses (including all uses in this group)
  // - Insert PHIs for any non-dominated uses
  // - Handle lane-aware SSA repair
  
  for (auto &G : Groups) {
    if (G.isDeleted())
      continue;
      
    MachineInstr *Head = G.getHead();
    LLVM_DEBUG(dbgs() << "\nEmitting reload for group head: " << *Head);
    LLVM_DEBUG(dbgs() << "  Group size: " << G.size() << " use(s)\n");
    
    // Emit reload before the group head
    // reloadBefore() will call MachineLaneSSAUpdater::repairSSAForNewDef()
    // which handles all SSA repair automatically
    Register NewVReg = reloadBefore(Head->getIterator(), SpilledVMP);
    
    LLVM_DEBUG(dbgs() << "  Reloaded into " << printReg(NewVReg, TRI) << "\n");
    LLVM_DEBUG(dbgs() << "  All " << G.size() 
                      << " use(s) in this group will be rewritten by SSA updater\n");
  }
  
  LLVM_DEBUG(dbgs() << "\nemitReloadsAndRepairSSA() complete: emitted " 
                    << NumGroups << " reload(s)\n");
  LLVM_DEBUG(dbgs() << "=================================\n\n");
}

// ============================================================================
// Primary spill/reload methods (forward iterators) - used for SSA repair
// ============================================================================

void AMDGPUSSARegisterSpiller::spillBefore(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertBefore,
    VRegMaskPair VMP) {
  Register VReg = VMP.getVReg();
  LaneBitmask Mask = VMP.getLaneMask();

  LLVM_DEBUG(dbgs() << "spillBefore(): Emitting spill for "
                    << printReg(VReg, TRI) << " with mask "
                    << PrintLaneMask(Mask) << "\n");

  // Get or create stack slot
  int FI = assignVirt2StackSlot(VMP);

  // Determine SubRegIdx from lane mask
  unsigned SubRegIdx = VMP.getSubReg(MRI, TRI);
  
  // Get the appropriate register class
  const TargetRegisterClass *RC =
      (SubRegIdx == AMDGPU::NoRegister)
          ? TRI->getRegClassForReg(*MRI, VReg)
          : TRI->getSubRegisterClass(TRI->getRegClassForReg(*MRI, VReg),
                                     SubRegIdx);

  LLVM_DEBUG({
    if (SubRegIdx != AMDGPU::NoRegister) {
      dbgs() << "spillBefore(): Spilling subregister "
             << TRI->getSubRegIndexName(SubRegIdx) << " of "
             << printReg(VReg, TRI) << "\n";
    }
  });

  // Emit the spill instruction
  bool IsKill = (SubRegIdx == AMDGPU::NoRegister);
  TII->storeRegToStackSlot(MBB, InsertBefore, VReg, IsKill, FI, RC, TRI, VReg,
                           MachineInstr::NoFlags, SubRegIdx);

  // Get the inserted spill instruction (it's right before InsertBefore)
  MachineInstr &SpillMI = *std::prev(InsertBefore);
  
  // Update LiveIntervals
  LIS->InsertMachineInstrInMaps(SpillMI);

  // NOTE: We do NOT trim LiveIntervals here!
  // The MachineLaneSSAUpdater will handle LiveInterval cleanup after
  // SSA repair is complete. Trimming here would create invalid MIR.

  LLVM_DEBUG(dbgs() << "spillBefore(): Emitted: " << SpillMI);
  ++NumSpills;
}

void AMDGPUSSARegisterSpiller::spillAtEnd(MachineBasicBlock &MBB,
                                          VRegMaskPair VMP) {
  spillBefore(MBB, MBB.getFirstTerminator(), VMP);
}

Register AMDGPUSSARegisterSpiller::reloadBefore(
    MachineBasicBlock::iterator InsertBefore, VRegMaskPair VMP) {
  MachineBasicBlock *MBB = InsertBefore->getParent();
  Register OrigVReg = VMP.getVReg();
  LaneBitmask Mask = VMP.getLaneMask();
  int FI = assignVirt2StackSlot(VMP);

  LLVM_DEBUG(dbgs() << "reloadBefore(): Emitting reload for "
                    << printReg(OrigVReg, TRI) << " with mask "
                    << PrintLaneMask(Mask) << "\n");

  // Get the appropriate register class for the reload
  const TargetRegisterClass *RC = VMP.getRegClass(MRI, TRI);

  // IMPORTANT: Emit reload that defines OrigVReg (violating SSA)
  // MachineLaneSSAUpdater will create a new VReg and repair SSA
  TII->loadRegFromStackSlot(*MBB, InsertBefore, OrigVReg, FI, RC, TRI, OrigVReg);

  // Get the reload instruction (it's right before InsertBefore)
  MachineInstr *ReloadMI = &*std::prev(InsertBefore);
  assert(ReloadMI && "Reload instruction not found");

  // Update LiveIntervals
  LIS->InsertMachineInstrInMaps(*ReloadMI);

  LLVM_DEBUG(dbgs() << "reloadBefore(): Emitted reload (violates SSA): " 
                    << *ReloadMI);
  LLVM_DEBUG(dbgs() << "reloadBefore(): MachineLaneSSAUpdater will repair SSA\n");

  // Call MachineLaneSSAUpdater to repair SSA form
  MachineFunction &MF = *MBB->getParent();
  MachineLaneSSAUpdater SSAUpdater(MF, *LIS, *DT, *TRI);
  Register NewVReg = SSAUpdater.repairSSAForNewDef(*ReloadMI, OrigVReg);

  LLVM_DEBUG(dbgs() << "reloadBefore(): SSA repaired, new register is "
                    << printReg(NewVReg, TRI) << "\n");

  ++NumReloads;
  return NewVReg;
}

Register AMDGPUSSARegisterSpiller::reloadAtEnd(MachineBasicBlock &MBB,
                                               VRegMaskPair VMP) {
  return reloadBefore(MBB.getFirstTerminator(), VMP);
}

// ============================================================================
// Backward traversal helper (reverse iterator) - used during spilling phase
// ============================================================================

void AMDGPUSSARegisterSpiller::spillBefore(
    MachineBasicBlock &MBB, MachineBasicBlock::reverse_iterator I,
    VRegMaskPair VMP) {
  // For reverse iterator during backward traversal:
  // We want to spill *after* the current instruction (before it in reverse order)
  // Convert reverse iterator to forward iterator using getReverse()
  MachineBasicBlock::iterator InsertPt = I.getReverse();
  spillBefore(MBB, InsertPt, VMP);
}

void AMDGPUSSARegisterSpiller::dumpRegSet(
    const VRegMaskPairSet &Regs) const {
  for (const auto &VMP : Regs) {
    dbgs() << "  " << printReg(VMP.getVReg(), TRI) << " (mask "
           << PrintLaneMask(VMP.getLaneMask()) << ", size "
           << VMP.getSizeInRegs(TRI) << ")\n";
  }
}

bool AMDGPUSSARegisterSpiller::runOnMachineFunction(MachineFunction &MF) {
  // Initialize pass dependencies
  TRI = static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  TII = static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MRI = &MF.getRegInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();
  FrameInfo = &MF.getFrameInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  Indexes = &getAnalysis<SlotIndexesWrapperPass>().getSI();
  DT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  // Get Next Use Analysis result
  NU = &getAnalysis<AMDGPUNextUseAnalysisWrapper>().getNU();

  LLVM_DEBUG(dbgs() << "AMDGPUSSARegisterSpiller: Processing function "
                    << MF.getName() << "\n");

  // Calculate register pressure limits based on subtarget and function requirements
  // These limits are determined by:
  // - Target architecture capabilities
  // - Desired occupancy (waves per execution unit)
  // - Function-specific requirements (e.g., flat scratch, dynamic stack)
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  
  // Get the maximum number of registers for this function
  // These methods account for:
  // - Hardware limits
  // - Addressable register limits
  // - Occupancy requirements
  // - Dynamic VGPR block size (for unified register file architectures)
  unsigned VGPRLimit = ST.getMaxNumVGPRs(MF);
  unsigned SGPRLimit = ST.getMaxNumSGPRs(MF);
  
  // Apply a safety margin to avoid over-spilling
  // We target slightly below the limit to leave room for:
  // - Register allocator's own requirements
  // - Compiler-generated temporaries
  // - ABI reserved registers
  // Using 90% of the limit as a conservative threshold
  VGPRLimit = (VGPRLimit * 9) / 10;
  SGPRLimit = (SGPRLimit * 9) / 10;

  LLVM_DEBUG(dbgs() << "Register pressure limits (90% of max): VGPR=" << VGPRLimit
                    << ", SGPR=" << SGPRLimit << "\n");
  LLVM_DEBUG(dbgs() << "  (Architecture max: VGPR=" << ST.getMaxNumVGPRs(MF)
                    << ", SGPR=" << ST.getMaxNumSGPRs(MF) << ")\n");

  // Two-pass approach:
  // Pass 1: Process SGPRs (spilled to VGPR lanes if needed)
  // Pass 2: Process VGPRs (spilled to memory)

  // Pass 1: SGPR Spilling
  LLVM_DEBUG(dbgs() << "\n=== Pass 1: Processing SGPRs ===\n");
  processFunction(MF, SGPRLimit, /*IsVGPRPass=*/false);

  // Pass 2: VGPR Spilling
  LLVM_DEBUG(dbgs() << "\n=== Pass 2: Processing VGPRs ===\n");
  processFunction(MF, VGPRLimit, /*IsVGPRPass=*/true);

  LLVM_DEBUG(dbgs() << "\nAMDGPUSSARegisterSpiller: Completed processing "
                    << MF.getName() << "\n");
  LLVM_DEBUG(dbgs() << "Total spills: " << NumSpills << ", Total reloads: "
                    << NumReloads << "\n");

  // Return true if we made any modifications
  return NumSpills > 0 || NumReloads > 0;
}


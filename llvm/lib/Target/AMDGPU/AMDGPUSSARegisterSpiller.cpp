//===--------------- AMDGPUSSARegisterSpiller.cpp  -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSSARegisterSpiller.h"
#include "AMDGPU.h"
#include "GCNRegPressure.h"
#include "GCNSubtarget.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/InitializePasses.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-spiller"

STATISTIC(NumSpills, "Number of register spills");
STATISTIC(NumReloads, "Number of register reloads");

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

    // Track spilled registers per block
    VRegMaskPairSet Spilled;

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

        // Call spill method with current RP and limit
        spill(*MBB, I, ActiveRegs, Spilled, CurRP, RPLimit);
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

unsigned AMDGPUSSARegisterSpiller::spill(MachineBasicBlock &MBB,
                                         MachineBasicBlock::reverse_iterator I,
                                         VRegMaskPairSet &Active,
                                         VRegMaskPairSet &Spilled,
                                         unsigned CurRP,
                                         unsigned RPLimit) {
  unsigned NumSpillsPerformed = 0;

  LLVM_DEBUG(dbgs() << "spill(): CurRP=" << CurRP << ", RPLimit=" << RPLimit << "\n");

  // Step 1: Calculate how much we need to spill
  if (CurRP <= RPLimit) {
    LLVM_DEBUG(dbgs() << "spill(): No spilling needed (RP <= limit)\n");
    return 0;
  }

  unsigned SizeToSpill = CurRP - RPLimit;
  LLVM_DEBUG(dbgs() << "spill(): Need to spill " << SizeToSpill
                    << " 32-bit register units\n");

  // Step 2: Sort Active set by next-use distance (longest last)
  sortRegSetByNextUse(MBB, I, Active);

  // Step 3: Greedily select registers to spill from the back
  VRegMaskPairSet ToSpill;
  unsigned RemainingToSpill = SizeToSpill;

  LLVM_DEBUG(dbgs() << "spill(): Need to reduce RP by " << RemainingToSpill
                    << " units\n");

  while (RemainingToSpill > 0 && !Active.empty()) {
    VRegMaskPair Candidate = Active.pop_back_val();
    unsigned CandidateSize = Candidate.getSizeInRegs(TRI);

    LLVM_DEBUG(dbgs() << "spill(): Considering candidate "
                      << printReg(Candidate.getVReg(), TRI) << " with mask "
                      << PrintLaneMask(Candidate.getLaneMask()) << " (size "
                      << CandidateSize << ")\n");

    // If this register is larger than what we need to spill, split it by
    // subregisters and only spill what's needed
    if (CandidateSize > RemainingToSpill) {
      LLVM_DEBUG(dbgs() << "spill(): Candidate is too large (" << CandidateSize
                        << " > " << RemainingToSpill
                        << "), splitting by subregisters\n");

      // Get subregisters sorted by next-use distance (longest first)
      // When traversing backward, we're always at a specific instruction
      MachineInstr *MI = &(*I);
      SmallVector<VRegMaskPair> SortedSubregs = NU->getSortedSubregUses(MI, Candidate);

      if (!SortedSubregs.empty()) {
        // Split by subregisters and spill only what's needed
        for (const auto &SubReg : SortedSubregs) {
          unsigned SubRegSize = SubReg.getSizeInRegs(TRI);
          
          if (!Spilled.contains(SubReg)) {
            ToSpill.insert(SubReg);
          }
          
          RemainingToSpill -= SubRegSize;
          
          if (RemainingToSpill == 0)
            break;
        }
        
        // Note: We don't need to insert remaining lanes back into Active.
        // RPTracker will provide the correct live set on the next instruction.
      } else {
        // Fallback: no subregister info available, spill the whole register
        LLVM_DEBUG(dbgs() << "spill(): No subregister info, spilling whole "
                             "register\n");
        if (!Spilled.contains(Candidate)) {
          ToSpill.insert(Candidate);
        }
        RemainingToSpill = 0;
      }
    } else {
      // This register fits within our spill budget
      if (!Spilled.contains(Candidate)) {
        ToSpill.insert(Candidate);
      }
      RemainingToSpill -= CandidateSize;
    }
  }

  // Step 5: Emit spill instructions for selected registers
  LLVM_DEBUG(dbgs() << "spill(): Emitting spill instructions for "
                    << ToSpill.size() << " register(s)\n");

  for (const auto &VMP : ToSpill) {
    LLVM_DEBUG(dbgs() << "spill(): Spilling " << printReg(VMP.getVReg(), TRI)
                      << " with mask " << PrintLaneMask(VMP.getLaneMask())
                      << "\n");

    spillBefore(MBB, I, VMP);
    Spilled.insert(VMP);
    NumSpillsPerformed++;
  }

  LLVM_DEBUG(dbgs() << "spill(): Final active set after spilling:\n";
             dumpRegSet(Active);
             dbgs() << "spill(): Final spilled set:\n"; dumpRegSet(Spilled));

  return NumSpillsPerformed;
}

void AMDGPUSSARegisterSpiller::spillBefore(MachineBasicBlock &MBB,
                                           MachineBasicBlock::reverse_iterator I,
                                           VRegMaskPair VMP) {
  // TODO: Implement spill instruction emission
  // Steps:
  // 1. Get stack slot: int FI = assignVirt2StackSlot(VMP);
  // 2. Get register class: const TargetRegisterClass *RC = VMP.getRegClass(MRI, TRI);
  // 3. Determine appropriate SI_SPILL_* opcode based on RC
  // 4. Convert reverse iterator to forward for insertion: auto FwdI = std::prev(I.base());
  // 5. Build spill instruction: TII->storeRegToStackSlot(MBB, FwdI, ...)
  // 6. Update LiveIntervals and SlotIndexes
  // 7. Increment NumSpills statistic

  LLVM_DEBUG(dbgs() << "spillBefore(): TODO - Emit spill for "
                    << printReg(VMP.getVReg(), TRI) << "\n");
}

void AMDGPUSSARegisterSpiller::reloadBefore(MachineBasicBlock &MBB,
                                            MachineBasicBlock::reverse_iterator I,
                                            VRegMaskPair VMP) {
  // TODO: Implement reload instruction emission and SSA repair
  // Steps:
  // 1. Get stack slot: int FI = assignVirt2StackSlot(VMP);
  // 2. Create new virtual register for reload result
  // 3. Emit reload instruction: TII->loadRegFromStackSlot(...)
  // 4. Use MachineLaneSSAUpdater::repairSSAForNewDef() to repair SSA:
  //    - Create MachineLaneSSAUpdater instance
  //    - Call repairSSAForNewDef(OrigVReg, NewVReg, ReloadMI, LanesToReload, ...)
  // 5. Update LiveIntervals and SlotIndexes
  // 6. Increment NumReloads statistic

  LLVM_DEBUG(dbgs() << "reloadBefore(): TODO - Emit reload for "
                    << printReg(VMP.getVReg(), TRI) << "\n");
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

  // TODO: Determine register pressure limits
  // For now, use placeholder values
  unsigned VGPRLimit = 24; // Example limit for VGPRs
  unsigned SGPRLimit = 80; // Example limit for SGPRs

  LLVM_DEBUG(dbgs() << "Register pressure limits: VGPR=" << VGPRLimit
                    << ", SGPR=" << SGPRLimit << "\n");

  // Two-pass approach:
  // Pass 1: Process SGPRs (spilled to VGPR lanes if needed)
  // Pass 2: Process VGPRs (spilled to memory)
  // This is necessary because SGPRs are 32-bit and can be spilled into
  // VGPR lanes (VGPRs are 64x32-bit lanes)

  // Pass 1: SGPR Spilling
  LLVM_DEBUG(dbgs() << "\n=== Pass 1: Processing SGPRs ===\n");
  processFunction(MF, SGPRLimit, /*IsVGPRPass=*/false);

  // Pass 2: VGPR Spilling
  LLVM_DEBUG(dbgs() << "\n=== Pass 2: Processing VGPRs ===\n");
  processFunction(MF, VGPRLimit, /*IsVGPRPass=*/true);

  // TODO: Implement reload insertion and SSA repair after both passes
  // For each spilled register:
  // 1. Find all uses that need reloads
  // 2. Insert reload instructions (for SGPRs: reload from VGPR lanes; for VGPRs: reload from memory)
  // 3. Use MachineLaneSSAUpdater to repair SSA form

  LLVM_DEBUG(dbgs() << "\nAMDGPUSSARegisterSpiller: Completed processing "
                    << MF.getName() << "\n");
  LLVM_DEBUG(dbgs() << "Total spills: " << NumSpills << ", Total reloads: "
                    << NumReloads << "\n");

  // Return true if we made any modifications
  return NumSpills > 0 || NumReloads > 0;
}


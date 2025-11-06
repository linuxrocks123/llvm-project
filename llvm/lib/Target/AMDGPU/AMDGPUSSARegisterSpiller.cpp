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
#include "llvm/Support/GenericIteratedDominanceFrontier.h"
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

static bool isReloadInstr(const MachineInstr *MI) {
  if (MI->getOpcode() == AMDGPU::SI_SPILL_S32_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S64_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S96_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S128_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S160_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S192_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S224_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S256_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S288_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S320_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S352_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S384_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S512_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_S1024_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V32_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V64_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V96_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V128_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V160_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V192_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V224_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V256_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V288_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V320_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V352_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V384_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V512_RESTORE ||
      MI->getOpcode() == AMDGPU::SI_SPILL_V1024_RESTORE)
    return true;
  return false;
}

/// Get register name for debug output - textual name if available, otherwise number
static std::string getRegNameForDebug(Register Reg, const MachineRegisterInfo *MRI,
                                       const TargetRegisterInfo *TRI) {
  if (!Reg.isValid())
    return "<invalid>";
  
  if (Reg.isVirtual()) {
    StringRef Name = MRI->getVRegName(Reg);
    if (!Name.empty())
      return ("%" + Name).str();
  }
  
  std::string Str;
  raw_string_ostream OS(Str);
  OS << printReg(Reg, TRI);
  return OS.str();
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

bool AMDGPUSSARegisterSpiller::processFunction(MachineFunction &MF,
                                               unsigned RPLimit,
                                               bool IsVGPRPass) {
  LLVM_DEBUG(dbgs() << "processFunction: " << (IsVGPRPass ? "VGPR" : "SGPR")
                    << " pass, limit=" << RPLimit << "\n");

  // Store RP limits for reload budget checking
  if (IsVGPRPass)
    VGPRLimit = RPLimit;
  else
    SGPRLimit = RPLimit;

  // Track if we made any modifications
  bool Changed = false;

  // Traverse basic blocks in reverse post-order (RPO)
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);

  for (MachineBasicBlock *MBB : RPOT) {
    LLVM_DEBUG(dbgs() << "\nProcessing " << printMBBReference(*MBB) << "\n");

    if (MBB->empty())
      continue;

    // Initialize GCNUpwardRPTracker - it analyzes backward to compute live state
    // But we walk the block forward (top-down) for spilling
    GCNUpwardRPTracker RPTracker(*LIS);
    
    // Traverse instructions forward (from beginning to end)
    // When we spill at point P, pressure drops from P forward
    for (auto I = MBB->begin(), E = MBB->end(); I != E; ++I) {
      MachineInstr &MI = *I;

      // Skip spill and reload instructions we create
      if (isSpillInstr(&MI) || isReloadInstr(&MI)) {
        LLVM_DEBUG(dbgs() << "  Skipping spill/reload: " << MI);
        continue;
      }

      LLVM_DEBUG(dbgs() << "  Processing: " << MI);

      // Reset tracker to this instruction - it will compute what's live here
      // by analyzing backward from this point
      RPTracker.reset(MI);

      // Get current register pressure at this point
      GCNRegPressure CurPressure = RPTracker.getPressure();
      
      // Get pressure for the current pass using the appropriate API
      const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
      unsigned CurRP = IsVGPRPass ? CurPressure.getVGPRNum(ST.hasGFX90AInsts())
                                  : CurPressure.getSGPRNum();

      LLVM_DEBUG(dbgs() << "    " << (IsVGPRPass ? "VGPR" : "SGPR")
                        << " pressure: " << CurRP << "\n");

      // Check if we need to spill
      if (CurRP > RPLimit) {
        LLVM_DEBUG(dbgs() << "  " << (IsVGPRPass ? "VGPR" : "SGPR")
                          << " pressure " << CurRP << " > limit " << RPLimit
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

        // CRITICAL: Exclude registers DEFINED by the current instruction!
        // RPTracker.reset(MI) gives us RP AFTER MI executes, which includes
        // registers defined by MI. But we insert spills BEFORE MI, so we cannot
        // spill a register that doesn't exist yet. We must exclude MI.defs().
        VRegMaskPairSet ToRemove;
        for (const MachineOperand &MO : MI.defs()) {
          if (MO.getReg().isVirtual()) {
            // Create VRegMaskPair from the def operand to match both reg and mask
            VRegMaskPair Def(MO, TRI, MRI);
            // Look for matching VMP in active set
            for (const auto &VMP : ActiveRegs) {
              if (Def == VMP) {
                ToRemove.insert(VMP);
                LLVM_DEBUG(dbgs() << "  Excluding " << printReg(Def.getVReg(), TRI) 
                                  << " with mask " << PrintLaneMask(Def.getLaneMask())
                                  << " (defined by current instruction)\n");
              }
            }
          }
        }
        // Remove them from active candidates
        for (const auto &VMP : ToRemove) {
          ActiveRegs.remove(VMP);
        }


        MachineBasicBlock::reverse_iterator ReverseI(std::next(I));
        
        // Call spillAndReload to handle atomic spill+reload+SSA repair
        bool Spilled = spillAndReload(*MBB, ReverseI, ActiveRegs, CurRP, RPLimit);
        if (Spilled) {
          Changed = true;
        }
        
        // Note: After spilling at point P, the spilled register's pressure
        // contribution is removed from P forward. We continue walking forward
        // and will see lower pressure at subsequent instructions.
      }
    }
  }
  return Changed;
}

void AMDGPUSSARegisterSpiller::sortRegSetByNextUse(
    MachineBasicBlock &MBB, MachineBasicBlock::reverse_iterator I,
    VRegMaskPairSet &Active) {
  // Pre-compute next-use distances for all registers to avoid redundant calls
  // during sorting (sort makes O(n log n) comparisons, but we only need O(n) distance calculations)
  DenseMap<VRegMaskPair, unsigned> DistanceMap;
  
  // Get the current instruction
  MachineInstr *MI = &(*I);
  
  // IMPORTANT: Query next-use distance AFTER current instruction executes
  // Example: X = ADD Y, Z
  //   At this instruction, Y and Z are inputs (used here)
  //   But AFTER this instruction, if Y and Z are dead, we can spill them to make room for X
  // 
  // reverse_iterator semantics: I.getReverse() points to position AFTER *I in forward order
  // So we query at I.getReverse() to get "next use after current instruction"
  //
  // Exception: For early-clobber DEFs (e.g., %X:early-clobber = OP %Y, %Z),
  // the output %X is written BEFORE inputs %Y, %Z are read.
  // Therefore, we CANNOT spill %Y or %Z to make room (they conflict).
  // Algorithm:
  //   1. Check if instruction has early-clobber DEF
  //   2. Collect USE operands (inputs that conflict with early-clobber output)
  //   3. Query all registers at NEXT position
  //   4. Mark conflicting USEs as distance 0 (cannot spill)
  // Note: DEF operands are NOT in Active set yet, so they'll never be candidates.
  
  // CRITICAL: I is reverse_iterator pointing to instruction A.
  // We want to query AFTER instruction A executes.
  // I.getReverse() returns I.base() which empirically points to instruction A itself.
  // To query AFTER instruction A, we need std::next(I.getReverse()).
  // BUT: std::next() might reach MBB.end(), which cannot be dereferenced.
  MachineBasicBlock::iterator AfterCurrent = std::next(I.getReverse());
  bool AtBlockEnd = (AfterCurrent == MBB.end());
  
  // Step 1: Check if instruction has early-clobber DEF
  bool HasEarlyClobber = false;
  for (const MachineOperand &MO : MI->operands()) {
    if (MO.isReg() && MO.isDef() && MO.isEarlyClobber()) {
      HasEarlyClobber = true;
      break;
    }
  }
  
  // Step 2: Collect USE operands (inputs) that conflict with early-clobber output
  VRegMaskPairSet EarlyClobberConflictingUses;
  if (HasEarlyClobber) {
    LLVM_DEBUG(dbgs() << "sortRegSetByNextUse: Early-clobber detected\n");
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isReg() && MO.isUse() && MO.getReg().isVirtual()) {
        VRegMaskPair VMP(MO, TRI, MRI);
        EarlyClobberConflictingUses.insert(VMP);
        
        LLVM_DEBUG(dbgs() << "  Conflicting USE: " << printReg(VMP.getVReg(), TRI);
                   if (MO.getSubReg())
                     dbgs() << "." << TRI->getSubRegIndexName(MO.getSubReg());
                   dbgs() << " (lanes: " << PrintLaneMask(VMP.getLaneMask()) << ")\n");
      }
    }
  }
  
  // Step 3: Calculate distance for each register
  for (const auto &VMP : Active) {
    unsigned Dist;
    
    // If at block end, use block-level query (checks successors)
    // Otherwise, query at the next instruction
    if (AtBlockEnd) {
      Dist = NU->getNextUseDistance(MBB, VMP);
    } else {
      Dist = NU->getNextUseDistance(AfterCurrent, VMP);
    }
    
    // Step 4: Mark early-clobber-conflicting USEs as cannot spill
    if (HasEarlyClobber) {
      LaneCoverageResult Coverage = EarlyClobberConflictingUses.getCoverage(VMP);
      if (!Coverage.isFullyUncovered()) {
        // This VMP overlaps with early-clobber USE operand(s), cannot spill
        Dist = 0; // Cannot spill (conflicts with early-clobber output)
        LLVM_DEBUG(dbgs() << "  " << printReg(VMP.getVReg(), TRI) 
                          << " (mask " << PrintLaneMask(VMP.getLaneMask()) << ")"
                          << " conflicts with early-clobber (covered: " 
                          << PrintLaneMask(Coverage.getCovered()) << "), dist=0\n");
      }
    }
    
    DistanceMap[VMP] = Dist;
  }
  
  // Sort using pre-computed distances
  Active.sort([&](const VRegMaskPair &A, const VRegMaskPair &B) {
    unsigned DistA = DistanceMap[A];
    unsigned DistB = DistanceMap[B];

    // Primary sort: Shorter distance first (longer distance at back for spilling)
    if (DistA != DistB)
      return DistA < DistB;

    // Tie-breaker: If distances are equal, prefer SMALLER register to spill
    // We pop from the back, so put LARGER registers first (smaller at back)
    // This ensures we spill exactly the amount needed, not more
    // Example: Need to free 2 VGPRs, both v64 and v128 have same distance
    //   → Put v128 first, v64 at back → pop v64 (2 VGPRs) instead of v128 (4 VGPRs)
    unsigned SizeA = A.getLaneMask().getNumLanes();
    unsigned SizeB = B.getLaneMask().getNumLanes();
    return SizeA > SizeB;  // Larger first, so smaller is at back for popping
  });
  
  LLVM_DEBUG({
    dbgs() << "sortRegSetByNextUse: Active set sorted at " << *MI;
    dbgs() << " (query position: after instruction";
    if (HasEarlyClobber)
      dbgs() << ", USE operands blocked from spilling";
    dbgs() << ")\n";
    
    for (const auto &VMP : Active) {
      Register VReg = VMP.getVReg();
      StringRef Name = MRI->getVRegName(VReg);
      if (!Name.empty())
        dbgs() << "  %" << Name;
      else
        dbgs() << "  " << printReg(VReg, TRI);
      dbgs() << " (mask " << PrintLaneMask(VMP.getLaneMask()) 
             << ") : " << DistanceMap[VMP];
      
      // Mark if this is a USE that conflicts with early-clobber
      if (HasEarlyClobber) {
        LaneCoverageResult Coverage = EarlyClobberConflictingUses.getCoverage(VMP);
        if (!Coverage.isFullyUncovered())
          dbgs() << " [blocked by early-clobber]";
      }
      dbgs() << "\n";
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

    LLVM_DEBUG({
      Register VReg = Candidate.getVReg();
      StringRef Name = MRI->getVRegName(VReg);
      dbgs() << "getVMPsToSpill(): Considering candidate ";
      if (!Name.empty())
        dbgs() << "%" << Name;
      else
        dbgs() << printReg(VReg, TRI);
      dbgs() << " with mask " << PrintLaneMask(Candidate.getLaneMask()) 
             << " (size " << CandidateSize << ")\n";
    });

    // If this register is larger than what we need to spill, split it by
    // subregisters and only spill what's needed
    if (CandidateSize > RemainingToSpill) {
      LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Candidate is too large (" 
                        << CandidateSize << " > " << RemainingToSpill
                        << "), splitting by subregisters\n");

      // Get subregisters sorted by next-use distance (longest first)
      // Use same query position as sortRegSetByNextUse: after instruction
      // Handle end-of-block case to avoid dereferencing end() iterator
      MachineBasicBlock::iterator AfterCurrent = std::next(I.getReverse());
      
      SmallVector<VRegMaskPair> SortedSubregs;
      if (AfterCurrent == MBB.end()) {
        // At block end, use block-level query
        SortedSubregs = NU->getSortedSubregUses(MBB, Candidate);
      } else {
        SortedSubregs = NU->getSortedSubregUses(AfterCurrent, Candidate);
      }

      if (!SortedSubregs.empty()) {
        // Split by subregisters and spill only what's needed
        for (const auto &SubReg : SortedSubregs) {
          unsigned SubRegSize = SubReg.getSizeInRegs(TRI);
          if (SubRegSize <= RemainingToSpill) {
            ToSpill.insert(SubReg);
            RemainingToSpill -= SubRegSize;
          
            if (RemainingToSpill == 0)
              break;
          } else {
            // for now let's spill larger subreg but take care of unsigned overflow
            ToSpill.insert(SubReg);
            RemainingToSpill = 0;
            break;
            // TODO: find sub_mask of size RemainingToSpill in SubReg and spill it
            /*LaneBitmask SubMask = SubReg.getLaneMask();
            LaneBitmask SpillMask = SubMask &
            LaneBitmask::getLow(RemainingToSpill);
            ToSpill.insert(VRegMaskPair(SubReg.getVReg(), SpillMask));
            RemainingToSpill -= SpillMask.getNumLanes();
            */
          }
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

    LLVM_DEBUG({
      StringRef Name = MRI->getVRegName(VReg);
      dbgs() << "\nspillAndReload(): Processing VMP ";
      if (!Name.empty())
        dbgs() << "%" << Name;
      else
        dbgs() << printReg(VReg, TRI);
      dbgs() << " with mask " << PrintLaneMask(Mask) << "\n";
    });

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
    emitReloadsAndRepairSSA(VMP, SpillMI, FI, &*I);
  }

  LLVM_DEBUG(dbgs() << "spillAndReload(): Completed, spilled " 
                    << ToSpill.size() << " VMP(s)\n");
  LLVM_DEBUG(dbgs() << "===================================\n\n");

  return true;
}

void AMDGPUSSARegisterSpiller::emitReloadsAndRepairSSA(
    VRegMaskPair SpilledVMP, MachineInstr *SpillMI, int FrameIndex,
    MachineInstr *TriggeringMI) {
  
  Register SpilledReg = SpilledVMP.getVReg();
  LaneBitmask SpilledMask = SpilledVMP.getLaneMask();
  
  LLVM_DEBUG({
    dbgs() << "\n=== emitReloadsAndRepairSSA() ===\n";
    StringRef Name = MRI->getVRegName(SpilledReg);
    dbgs() << "Spilled: ";
    if (!Name.empty())
      dbgs() << "%" << Name;
    else
      dbgs() << printReg(SpilledReg, TRI);
    dbgs() << " with mask " << PrintLaneMask(SpilledMask) << "\n";
    dbgs() << "SpillMI: " << *SpillMI;
  });

  // Step 1: Classify uses into dominated and non-dominated
  // Defer IDF calculation until we know we need it (optimization for common case)
  SmallVector<MachineInstr *> DominatedUses;
  SmallVector<MachineInstr *> NonDominatedUses;
  
  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(SpilledReg)) {
    // Skip the spill instruction itself
    if (&UseMI == SpillMI)
      continue;
    
    // Skip the triggering instruction (that caused the spill)
    // This instruction's operands are live BEFORE it executes, so the spill
    // happens BEFORE this instruction reads the register. We should NOT
    // consider this a dominated use requiring reload.
    if (&UseMI == TriggeringMI)
      continue;
    
    // Skip other spill instructions (they're not "real" uses we want to reload for)
    // This primarily handles previous spills of overlapping subregisters.
    // Note: Reload instructions will never appear here - they *define* the register,
    // they don't *use* it, so they won't be in the use list.
    if (isSpillInstr(&UseMI))
      continue;
    
    // Skip PHI instructions - they're SSA repair artifacts, not real uses
    // PHI uses are handled automatically by MachineLaneSSAUpdater's SSA repair.
    // If we try to manually handle them with block splitting, we create an infinite loop:
    //   1. Split block and insert reload PHI
    //   2. SSAUpdater creates more PHIs
    //   3. These PHIs use the spilled register
    //   4. Next iteration finds these PHI uses and splits again
    //   5. Infinite loop!
    // 
    // Note: This is safe even for original PHIs in the input program because:
    //   - If PHI is dominated by spill: handled by normal reload+rewrite
    //   - If PHI is not dominated: SSAUpdater will insert merge PHIs automatically
    if (UseMI.isPHI())
      continue;
    
    // Check if this use actually uses the spilled lanes
    // For subregister spills, we only reload for uses that overlap with spilled mask
    bool UsesSpilledLanes = false;
    for (const MachineOperand &MO : UseMI.uses()) {
      if (MO.isReg() && MO.getReg() == SpilledReg) {
        LaneBitmask UseMask = TRI->getSubRegIndexLaneMask(MO.getSubReg());
        if (UseMask == LaneBitmask::getAll())
          UseMask = MRI->getMaxLaneMaskForVReg(SpilledReg);
        // Check if this use overlaps with the spilled lanes
        if ((UseMask & SpilledMask).any()) {
          UsesSpilledLanes = true;
          break;
        }
      }
    }
    
    if (!UsesSpilledLanes)
      continue;
    
    // Classify: dominated or not
    if (DT->dominates(SpillMI, &UseMI)) {
      DominatedUses.push_back(&UseMI);
      LLVM_DEBUG(dbgs() << "  Dominated use: " << UseMI);
    } else {
      NonDominatedUses.push_back(&UseMI);
      LLVM_DEBUG(dbgs() << "  Non-dominated use: " << UseMI);
    }
  }
  
  // Step 1a: Calculate IDF only if we have non-dominated uses
  // Then classify non-dominated uses as reachable or unreachable
  SmallVector<MachineInstr *> ReachableUses;
  SmallVector<MachineBasicBlock *> PrunedIDFBlocks;
  
  if (!NonDominatedUses.empty()) {
    MachineBasicBlock *SpillBlock = SpillMI->getParent();
    SmallPtrSet<MachineBasicBlock *, 4> DefBlocks;
    DefBlocks.insert(SpillBlock);
    
    IDFCalculatorBase<MachineBasicBlock, false> IDF(*DT);
    SmallVector<MachineBasicBlock *> IDFBlocks;
    IDF.setDefiningBlocks(DefBlocks);
    IDF.calculate(IDFBlocks);
    for (auto *BB : IDFBlocks) {
      // Check if SpilledReg is live-in to this block
      bool IsLiveIn = false;
      if (LIS) {
        LiveInterval &LI = LIS->getInterval(SpilledReg);
        SlotIndex BlockStart = LIS->getMBBStartIdx(BB);
        // Check if the register is live at block start
        if (LI.liveAt(BlockStart)) {
          IsLiveIn = true;
        }
      }
      
      if (IsLiveIn) {
        PrunedIDFBlocks.push_back(BB);
      }
    }
    
    LLVM_DEBUG({
      dbgs() << "  IDF blocks for spill (pruned by LiveIn): ";
      for (auto *BB : PrunedIDFBlocks)
        dbgs() << BB->getName() << " ";
      dbgs() << "\n";
    });
    
    // Classify non-dominated uses: reachable or unreachable
    for (MachineInstr *UseMI : NonDominatedUses) {
      MachineBasicBlock *UseBB = UseMI->getParent();
      
      // Use is reachable if:
      // 1. Use is in a pruned IDF block, OR
      // 2. Use is dominated by a pruned IDF block
      bool IsReachable = false;
      
      if (llvm::find(PrunedIDFBlocks, UseBB) != PrunedIDFBlocks.end()) {
        IsReachable = true;
      } else {
        for (auto *IDFBB : PrunedIDFBlocks) {
          if (DT->dominates(IDFBB, UseBB)) {
            IsReachable = true;
            break;
          }
        }
      }
      
      if (IsReachable) {
        ReachableUses.push_back(UseMI);
        LLVM_DEBUG(dbgs() << "  Classified as reachable (divergent): " << *UseMI);
      } else {
        LLVM_DEBUG(dbgs() << "  Unreachable use (ignored): " << *UseMI);
      }
    }
  }
  
  LLVM_DEBUG(dbgs() << "Found " << DominatedUses.size() 
                    << " dominated use(s), " << ReachableUses.size()
                    << " reachable use(s)\n");
  
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

    // Check if head was already rewritten by previous group's SSA repair
    // This can happen when SSA updater inserts PHIs that dominate later groups
    if (!usesSpilledVMP(Head, SpilledVMP)) {
      LLVM_DEBUG(dbgs() << "\nSkipping group (head already rewritten): " << *Head);
      continue;
    }

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
  
  // Step 4: Handle reachable but not dominated uses (divergent paths)
  if (!ReachableUses.empty()) {
    LLVM_DEBUG(dbgs() << "\n=== Handling " << ReachableUses.size() 
                      << " reachable (divergent path) uses ===\n");
    
    // Step 4a: Try to hoist spill to NCD if no uses on either path
    bool Hoisted = tryHoistSpillToNCD(SpillMI, SpilledVMP, ReachableUses);
    
    if (Hoisted) {
      LLVM_DEBUG(dbgs() << "  Spill hoisted to NCD, uses now dominated\n");
      LLVM_DEBUG(dbgs() << "  Reloads will be handled by standard dominated use logic\n");
      // After hoisting, uses are now dominated, so they'll be handled by
      // the standard reload placement in the next spilling iteration
      return;
    }
    
    // Step 4b: Hoisting impossible, classify and handle each use
    for (MachineInstr *UseMI : ReachableUses) {
      // Check if this use was already rewritten by SSA repair
      // Can happen from: (1) dominated use SSA repair, or (2) previous reachable use SSA repair
      if (!usesSpilledVMP(UseMI, SpilledVMP)) {
        LLVM_DEBUG(dbgs() << "  Skipping use (already rewritten): " << *UseMI);
        continue;
      }

      LLVM_DEBUG(dbgs() << "  Processing reachable use: " << *UseMI);
      
      // Emit reload instruction (no SSA repair yet - will be done after CFG transform)
      MachineInstr *ReloadMI = emitReload(UseMI->getIterator(), SpilledVMP);
      
      MachineBasicBlock *UseBB = UseMI->getParent();
      
      // Classify as uniform or divergent:
      // Uniform iff BOTH paths are non-divergent:
      //   1. NCD(Spill,Use) → SpillMI has no EXEC write or VCC branch
      //   2. NCD(Spill,Use) → UseMI has no EXEC write or VCC branch
      // Use optimized single-DFS check for diamond CFG
      
      MachineBasicBlock *SpillBB = SpillMI->getParent();
      MachineBasicBlock *NCD = DT->findNearestCommonDominator(SpillBB, UseBB);
      
      // Single DFS checking both paths simultaneously
      bool IsDivergent = pathsHaveDivergence(NCD, SpillMI, UseMI);
      
      LLVM_DEBUG({
        dbgs() << "    NCD: " << printMBBReference(*NCD) << "\n";
        dbgs() << "    Classification: " << (IsDivergent ? "DIVERGENT" : "UNIFORM") << "\n";
      });
      
      if (IsDivergent) {
        // Divergent: EXEC/VCC changes between NCD and spill/use
        handleDivergentReachableUse(SpillMI, ReloadMI, SpilledVMP, PrunedIDFBlocks);
      } else {
        // Uniform: EXEC/VCC stable on both paths
        handleUniformReachableUse(SpillMI, ReloadMI, SpilledVMP, PrunedIDFBlocks);
      }
      
      // Now repair SSA with the transformed CFG
      Register NewVReg = repairSSAForReload(ReloadMI, SpilledVMP);
      
      LLVM_DEBUG(dbgs() << "    SSA repair complete for reload, new register is "
                        << getRegNameForDebug(NewVReg, MRI, TRI) << "\n");
      
      // Store spill->reload mapping for potential future optimization
      SpillToReloadMap[SpillMI].push_back(ReloadMI);
    }
    
    LLVM_DEBUG(dbgs() << "  Processed " << ReachableUses.size() 
                      << " reachable use(s)\n");
  }
  
  // After all SSA repairs, shrink the original spilled register's LiveInterval
  // The SSAUpdater has rewritten uses to point to reloaded registers, so the
  // original register now has fewer live ranges and should contribute less to
  // register pressure. This is critical for the RPTracker to see correct pressure.
  LiveInterval &SpilledLI = LIS->getInterval(SpilledReg);
  LIS->shrinkToUses(&SpilledLI);
  
  LLVM_DEBUG(dbgs() << "\nemitReloadsAndRepairSSA() complete: emitted " 
                    << NumGroups << " reload(s) for dominated uses, "
                    << ReachableUses.size() << " reload(s) for reachable uses\n");
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
  
  LLVM_DEBUG({
    StringRef Name = MRI->getVRegName(VReg);
    dbgs() << "spillBefore(): Emitting spill for ";
    if (!Name.empty())
      dbgs() << "%" << Name;
    else
      dbgs() << printReg(VReg, TRI);
    dbgs() << " with mask " << PrintLaneMask(Mask) << "\n";
  });

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
      StringRef Name = MRI->getVRegName(VReg);
      dbgs() << "spillBefore(): Spilling subregister "
             << TRI->getSubRegIndexName(SubRegIdx) << " of ";
      if (!Name.empty())
        dbgs() << "%" << Name;
      else
        dbgs() << printReg(VReg, TRI);
      dbgs() << "\n";
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

  // NOTE: We do NOT call shrinkToUses here because the original uses still exist.
  // After emitReloadsAndRepairSSA() rewrites uses to point to reloaded registers,
  // we'll shrink the LiveInterval there to reflect the reduced liveness.

  LLVM_DEBUG(dbgs() << "spillBefore(): Emitted: " << SpillMI);
  ++NumSpills;
}

void AMDGPUSSARegisterSpiller::spillAtEnd(MachineBasicBlock &MBB,
                                          VRegMaskPair VMP) {
  spillBefore(MBB, MBB.getFirstTerminator(), VMP);
}

MachineInstr *AMDGPUSSARegisterSpiller::emitReload(
    MachineBasicBlock::iterator InsertBefore, VRegMaskPair VMP) {
  MachineBasicBlock *MBB = InsertBefore->getParent();
  Register OrigVReg = VMP.getVReg();
  LaneBitmask Mask = VMP.getLaneMask();
  int FI = assignVirt2StackSlot(VMP);

  LLVM_DEBUG(dbgs() << "emitReload(): Emitting reload for "
                    << getRegNameForDebug(OrigVReg, MRI, TRI)
                    << " with mask " << PrintLaneMask(Mask) << "\n");

  // Determine SubRegIdx from lane mask
  unsigned SubRegIdx = VMP.getSubReg(MRI, TRI);
  
  // Get register class for the reload
  // For subreg reloads, RC is the subregister class
  // For full register reloads, RC is the original register class
  const TargetRegisterClass *RC = VMP.getRegClass(MRI, TRI);

  LLVM_DEBUG({
    if (SubRegIdx != AMDGPU::NoRegister) {
      dbgs() << "emitReload(): Subregister reload - reloading to "
             << getRegNameForDebug(OrigVReg, MRI, TRI) << "."
             << TRI->getSubRegIndexName(SubRegIdx)
             << " (class " << TRI->getRegClassName(RC) << ")\n";
    } else {
      dbgs() << "emitReload(): Full register reload to "
             << getRegNameForDebug(OrigVReg, MRI, TRI)
             << " (class " << TRI->getRegClassName(RC) << ")\n";
    }
  });

  // CHECK: Verify current RP is within budget before reload
  // After reload, SSA updater will replace uses of the old register,
  // making it dead. Net RP change is ~0 (reload replaces old register).
  // If RP is already over budget, reloading is impossible!
  
  // Get live registers at the insertion point
  GCNRPTracker::LiveRegSet LiveRegs = getLiveRegsBefore(*InsertBefore, *LIS);
  
  // Compute current pressure
  GCNRegPressure CurPressure = getRegPressure(*MRI, LiveRegs);
  const GCNSubtarget &ST = MBB->getParent()->getSubtarget<GCNSubtarget>();
  
  // Get current pressure for the appropriate register class
  unsigned CurRP = 0;
  unsigned RPLimitForCheck = 0;
  
  if (TRI->isVGPRClass(RC)) {
    CurRP = CurPressure.getVGPRNum(ST.hasGFX90AInsts());
    RPLimitForCheck = VGPRLimit;
  } else if (TRI->isSGPRClass(RC)) {
    CurRP = CurPressure.getSGPRNum();
    RPLimitForCheck = SGPRLimit;
  }
  
  LLVM_DEBUG(dbgs() << "emitReload(): RP check: current=" << CurRP
                    << " (limit=" << RPLimitForCheck << ")\n");
  
  if (CurRP > RPLimitForCheck) {
    std::string Msg;
    raw_string_ostream OS(Msg);
    OS << "SSA Spiller: Cannot reload register " 
       << getRegNameForDebug(OrigVReg, MRI, TRI)
       << " - current register pressure already exceeds budget!\n";
    OS << "  Current RP: " << CurRP << "\n";
    OS << "  Budget limit: " << RPLimitForCheck << "\n";
    OS << "  Note: Reload will replace old register use, but RP will remain at " << CurRP << "\n";
    OS << "  This indicates the input program requires more registers than available.\n";
    OS << "  Insertion point: " << *InsertBefore;
    OS << "  Block: " << printMBBReference(*MBB) << "\n";
    report_fatal_error(Twine(OS.str()));
  }

  // Emit reload instruction directly to OrigVReg (or OrigVReg.subreg)
  // This temporarily violates SSA, which MachineLaneSSAUpdater will fix
  TII->loadRegFromStackSlot(*MBB, InsertBefore, OrigVReg, FI, RC, TRI, 
                            Register(), MachineInstr::NoFlags, SubRegIdx);

  // Get the reload instruction (it's right before InsertBefore)
  MachineInstr *ReloadMI = &*std::prev(InsertBefore);
  assert(ReloadMI && "Reload instruction not found");

  // Update LiveIntervals
  LIS->InsertMachineInstrInMaps(*ReloadMI);

  LLVM_DEBUG(dbgs() << "emitReload(): Emitted reload: " << *ReloadMI);

  ++NumReloads;
  return ReloadMI;
}

Register AMDGPUSSARegisterSpiller::repairSSAForReload(
    MachineInstr *ReloadMI, VRegMaskPair VMP) {
  Register OrigVReg = VMP.getVReg();
  MachineBasicBlock *MBB = ReloadMI->getParent();
  
  LLVM_DEBUG(dbgs() << "repairSSAForReload(): Repairing SSA for reload: " 
                    << *ReloadMI);

  // Call MachineLaneSSAUpdater to repair SSA form
  // It will create a new SSA register with the appropriate class based on DefMask
  MachineFunction &MF = *MBB->getParent();
  MachineLaneSSAUpdater SSAUpdater(MF, *LIS, *DT, *TRI);
  Register NewVReg = SSAUpdater.repairSSAForNewDef(*ReloadMI, OrigVReg);

  LLVM_DEBUG(dbgs() << "repairSSAForReload(): SSA repaired, new register is "
                    << printReg(NewVReg, TRI);
             if (NewVReg.isVirtual()) {
               StringRef Name = MRI->getVRegName(NewVReg);
               if (!Name.empty())
                 dbgs() << " (%" << Name << ")";
             }
             dbgs() << "\n");

  return NewVReg;
}

Register AMDGPUSSARegisterSpiller::reloadBefore(
    MachineBasicBlock::iterator InsertBefore, VRegMaskPair VMP) {
  // Convenience wrapper: emit reload + repair SSA immediately
  MachineInstr *ReloadMI = emitReload(InsertBefore, VMP);
  return repairSSAForReload(ReloadMI, VMP);
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
  // Convert reverse iterator to forward iterator
  // The reverse iterator I is set up so that *I is the instruction MI where
  // we detected high pressure. We want to insert the spill BEFORE MI.
  // Simply get the forward iterator directly from the instruction.
  MachineInstr *MI = &(*I);
  MachineBasicBlock::iterator InsertPt = MI->getIterator();
  
  // If we're at a terminator, insert before the first terminator
  // to keep the block valid (terminators must be at the end)
  if (InsertPt != MBB.end() && InsertPt->isTerminator()) {
    InsertPt = MBB.getFirstTerminator();
  }
  
  spillBefore(MBB, InsertPt, VMP);
}

void AMDGPUSSARegisterSpiller::dumpRegSet(
    const VRegMaskPairSet &Regs) const {
  for (const auto &VMP : Regs) {
    Register VReg = VMP.getVReg();
    dbgs() << "  ";
    
    // Print original name if available (e.g., %large), otherwise print number
    StringRef Name = MRI->getVRegName(VReg);
    if (!Name.empty())
      dbgs() << "%" << Name;
    else
      dbgs() << printReg(VReg, TRI);
    
    dbgs() << " (mask " << PrintLaneMask(VMP.getLaneMask()) 
           << ", size " << VMP.getSizeInRegs(TRI) << ")\n";
  }
}

// ============================================================================
// Divergent Path Optimization Helpers
// ============================================================================

bool AMDGPUSSARegisterSpiller::usesSpilledVMP(const MachineInstr *MI, 
                                               VRegMaskPair SpilledVMP) const {
  Register SpilledReg = SpilledVMP.getVReg();
  LaneBitmask SpilledMask = SpilledVMP.getLaneMask();
  
  // Quick check: does the instruction read this virtual register at all?
  // This handles partial defines correctly (read-modify-write)
  if (!MI->readsVirtualRegister(SpilledReg))
    return false;
  
  // Found a use, now check if it overlaps with spilled lanes
  for (const MachineOperand &MO : MI->uses()) {
    if (MO.isReg() && MO.getReg() == SpilledReg) {
      LaneBitmask UseMask = TRI->getSubRegIndexLaneMask(MO.getSubReg());
      if (UseMask == LaneBitmask::getAll())
        UseMask = MRI->getMaxLaneMaskForVReg(SpilledReg);
      // Check if this use overlaps with the spilled lanes
      if ((UseMask & SpilledMask).any()) {
        return true;
      }
    }
  }
  
  return false;
}

bool AMDGPUSSARegisterSpiller::isDivergentInstr(const MachineInstr *MI) const {
  // Check for EXEC modifications
  if (MI->modifiesRegister(AMDGPU::EXEC, TRI) || 
      MI->modifiesRegister(AMDGPU::EXEC_LO, TRI))
    return true;
  
  // Check for VCC branches (vector conditional branches cause divergence)
  // VCC branches are typically S_CBRANCH_VCCZ, S_CBRANCH_VCCNZ
  if (MI->isBranch()) {
    for (const MachineOperand &MO : MI->uses()) {
      if (MO.isReg() && 
          (MO.getReg() == AMDGPU::VCC || 
           MO.getReg() == AMDGPU::VCC_LO ||
           MO.getReg().isVirtual())) {
        // Conservatively assume virtual register branches could be VCC-based
        return true;
      }
    }
  }
  
  return false;
}

bool AMDGPUSSARegisterSpiller::pathHasDivergence(
    MachineBasicBlock *StartBB, MachineInstr *StopInstr) const {
  
  // TODO: Implement caching - store results in DenseMap<(BB, Instr), bool>
  // to avoid redundant DFS traversals for the same queries
  
  // NOTE: StartBB is EXCLUDED from the search (we start from its successors)
  // This is used when StartBB is NCD - we check from NCD's exit to StopInstr
  
  MachineBasicBlock *EndBB = StopInstr->getParent();
  
  // DFS from StartBB's successors to StopInstr, checking for divergence
  SmallPtrSet<MachineBasicBlock *, 8> Visited;
  SmallVector<MachineBasicBlock *, 8> Worklist(StartBB->successors());
  
  while (!Worklist.empty()) {
    MachineBasicBlock *BB = Worklist.pop_back_val();
    
    if (!Visited.insert(BB).second)
      continue;
      
    // Check instructions in this block
    // If this is EndBB, only check up to StopInstr
    auto EndIt = (BB == EndBB) ? StopInstr->getIterator() : BB->end();
    
    for (auto I = BB->begin(), E = EndIt; I != E; ++I) {
      if (isDivergentInstr(&*I)) {
        LLVM_DEBUG(dbgs() << "  Found divergent instruction on path: " << *I);
        return true;
      }
    }
    
    // If we reached the end block, stop
    if (BB == EndBB)
      continue;
      
    // Continue to successors
    for (MachineBasicBlock *Succ : BB->successors()) {
      if (!Visited.count(Succ))
        Worklist.push_back(Succ);
    }
  }
  
  return false;
}

bool AMDGPUSSARegisterSpiller::pathsHaveDivergence(
    MachineBasicBlock *NCD, MachineInstr *SpillMI, MachineInstr *UseMI) const {
  
  // Check if any path from NCD to UseMI contains divergent instruction.
  // SpillMI is just context - we're checking if reload path has divergence.
  // Single DFS from NCD to UseMI with shared visited set for efficiency.
  
  (void)SpillMI; // Unused - just for context/symmetry in API
  MachineBasicBlock *UseBB = UseMI->getParent();
  
  // DFS from NCD's successors, checking all paths until we reach UseMI
  SmallPtrSet<MachineBasicBlock *, 8> Visited;
  SmallVector<MachineBasicBlock *, 8> Worklist(NCD->successors());
  
  while (!Worklist.empty()) {
    MachineBasicBlock *BB = Worklist.pop_back_val();
    
    if (!Visited.insert(BB).second)
      continue;
      
    // Determine iteration bounds for this block
    // If this is UseBB, only check up to UseMI (exclusive)
    auto EndIt = (BB == UseBB) ? UseMI->getIterator() : BB->end();
    
    // Check for divergent instructions in this block
    for (auto I = BB->begin(), E = EndIt; I != E; ++I) {
      if (isDivergentInstr(&*I)) {
        LLVM_DEBUG(dbgs() << "  Found divergent instruction on path: " << *I);
        return true;
      }
    }
    
    // Stop exploring if we reached UseMI
    if (BB == UseBB)
      continue;
      
    // Continue to successors
    for (MachineBasicBlock *Succ : BB->successors()) {
      if (!Visited.count(Succ))
        Worklist.push_back(Succ);
    }
  }
  
  return false;
}

bool AMDGPUSSARegisterSpiller::hasUseOnPath(
    MachineBasicBlock *StartBB, MachineBasicBlock *EndBB, 
    VRegMaskPair SpilledVMP, MachineInstr *StopInstr) const {
  
  // TODO: Implement caching - store results in DenseMap<(BB1, BB2, VMP), bool>
  // to avoid redundant DFS traversals for the same queries
  
  // NOTE: StartBB is EXCLUDED from the search (we start from its successors)
  // This is used when StartBB is NCD - we don't want to consider uses in NCD itself,
  // only uses between NCD's exit and EndBB (or before StopInstr in EndBB)
  
  Register VReg = SpilledVMP.getVReg();
  LaneBitmask Mask = SpilledVMP.getLaneMask();
  
  // DFS from StartBB's successors to EndBB, checking for uses
  SmallPtrSet<MachineBasicBlock *, 8> Visited;
  SmallVector<MachineBasicBlock *, 8> Worklist(StartBB->successors());
  
  while (!Worklist.empty()) {
    MachineBasicBlock *BB = Worklist.pop_back_val();
    
    if (!Visited.insert(BB).second)
      continue;
      
    // Check for uses in this block
    // If this is EndBB and StopInstr is specified, only check up to StopInstr
    auto EndIt = (BB == EndBB && StopInstr) ? StopInstr->getIterator() : BB->end();
    
    for (auto I = BB->begin(), E = EndIt; I != E; ++I) {
      const MachineInstr &MI = *I;
      
      for (const MachineOperand &MO : MI.uses()) {
        if (MO.isReg() && MO.getReg() == VReg) {
          // Check lane mask overlap
          LaneBitmask UseMask = TRI->getSubRegIndexLaneMask(MO.getSubReg());
          if (UseMask == LaneBitmask::getAll())
            UseMask = MRI->getMaxLaneMaskForVReg(VReg);
          
          if ((UseMask & Mask).any()) {
            LLVM_DEBUG(dbgs() << "  Found use on path at: " << MI);
            return true;
          }
        }
      }
    }
    
    // If we reached the end block, stop
    if (BB == EndBB)
      continue;
      
    // Continue to successors
    for (MachineBasicBlock *Succ : BB->successors()) {
      if (!Visited.count(Succ))
        Worklist.push_back(Succ);
    }
  }
  
  return false;
}

bool AMDGPUSSARegisterSpiller::tryHoistSpillToNCD(
    MachineInstr *SpillMI, VRegMaskPair SpilledVMP,
    const SmallVectorImpl<MachineInstr *> &ReachableUses) {
  
  if (ReachableUses.empty())
    return false;
    
  MachineBasicBlock *SpillBB = SpillMI->getParent();
  
  // Find nearest common dominator (NCD) of spill and all reachable uses
  MachineBasicBlock *NCD = SpillBB;
  for (MachineInstr *UseMI : ReachableUses) {
    MachineBasicBlock *UseBB = UseMI->getParent();
    NCD = DT->findNearestCommonDominator(NCD, UseBB);
    if (!NCD) {
      LLVM_DEBUG(dbgs() << "  No common dominator found, cannot hoist\n");
      return false;
    }
  }
  
  LLVM_DEBUG(dbgs() << "  NCD for hoisting: " << printMBBReference(*NCD) << "\n");
  
  // Check if there are uses on path from NCD to SpillMI (exclusive)
  // This checks blocks between NCD and SpillBB, plus SpillBB up to (but not including) SpillMI
  if (hasUseOnPath(NCD, SpillBB, SpilledVMP, SpillMI)) {
    LLVM_DEBUG(dbgs() << "  Uses exist on NCD→Spill path, cannot hoist\n");
    return false;
  }
  
  // Check paths from NCD to each reachable use (the "clean" paths)
  // Stop at the use instruction itself (don't include it)
  for (MachineInstr *UseMI : ReachableUses) {
    MachineBasicBlock *UseBB = UseMI->getParent();
    if (UseBB != SpillBB && hasUseOnPath(NCD, UseBB, SpilledVMP, UseMI)) {
      LLVM_DEBUG(dbgs() << "  Uses exist on NCD→Use path, cannot hoist\n");
      return false;
    }
  }
  
  LLVM_DEBUG(dbgs() << "  No uses on either path, hoisting spill to NCD\n");
  
  // Hoist: Move spill to end of NCD (before terminator)
  SpillMI->removeFromParent();
  NCD->insert(NCD->getFirstTerminator(), SpillMI);
  
  // Update LiveIntervals for the moved instruction
  LIS->handleMove(*SpillMI);
  
  LLVM_DEBUG(dbgs() << "  Hoisted spill to: " << printMBBReference(*NCD) << "\n");
  
  return true;
}

MachineInstr* AMDGPUSSARegisterSpiller::splitBlockBeforeReload(
    MachineInstr *SpillMI, MachineInstr *ReloadMI, VRegMaskPair SpilledVMP,
    const SmallVectorImpl<MachineBasicBlock *> &IDFBlocks) {
  
  LLVM_DEBUG(dbgs() << "    splitBlockBeforeReload() - creating conditional reload CFG\n");
  
  MachineBasicBlock *SpillBB = SpillMI->getParent();
  MachineBasicBlock *UseBB = ReloadMI->getParent();
  MachineFunction *MF = SpillBB->getParent();
  
  LLVM_DEBUG(dbgs() << "      SpillBB: " << printMBBReference(*SpillBB) << "\n");
  LLVM_DEBUG(dbgs() << "      UseBB: " << printMBBReference(*UseBB) << "\n");
  
  // Step 1: Find JoinBB - IDF block that dominates UseBB
  // (IDF blocks are where spill and clean paths merge)
  MachineBasicBlock *JoinBB = nullptr;
  for (MachineBasicBlock *IDF : IDFBlocks) {
    if (DT->dominates(IDF, UseBB)) {
      JoinBB = IDF;
      break;
    }
  }
  
  if (!JoinBB) {
    LLVM_DEBUG(dbgs() << "      WARNING: No JoinBB found, UseBB may BE the join\n");
    // If UseBB has multiple predecessors and is in IDF, it IS the join
    if (UseBB->pred_size() > 1 && 
        std::find(IDFBlocks.begin(), IDFBlocks.end(), UseBB) != IDFBlocks.end()) {
      JoinBB = UseBB;
    } else {
      LLVM_DEBUG(dbgs() << "      ERROR: Cannot determine JoinBB for split\n");
      llvm_unreachable("Cannot determine JoinBB for conditional reload");
    }
  }
  
  LLVM_DEBUG(dbgs() << "      JoinBB: " << printMBBReference(*JoinBB) << "\n");
  
  // Step 2: Insert boolean flag PHI at JoinBB
  // Flag = PHI(1 from spill-path, 0 from clean-paths)
  // First, materialize constants in each predecessor
  DenseMap<MachineBasicBlock *, Register> PredToFlagReg;
  
  for (MachineBasicBlock *Pred : JoinBB->predecessors()) {
    bool IsSpillPath = (Pred == SpillBB) || DT->dominates(SpillBB, Pred);
    Register ConstReg = MRI->createVirtualRegister(&AMDGPU::SReg_32RegClass);
    
    // Insert S_MOV_B32 at end of predecessor (before terminator)
    auto InsertPt = Pred->getFirstTerminator();
    MachineInstr *MovMI = BuildMI(*Pred, InsertPt, SpillMI->getDebugLoc(),
                                   TII->get(AMDGPU::S_MOV_B32), ConstReg)
                              .addImm(IsSpillPath ? 1 : 0);
    LIS->InsertMachineInstrInMaps(*MovMI);
    
    PredToFlagReg[Pred] = ConstReg;
    LLVM_DEBUG(dbgs() << "      Materialized constant " << (IsSpillPath ? 1 : 0) 
                      << " in " << printMBBReference(*Pred) << "\n");
  }
  
  // Now create PHI merging the flags
  Register FlagReg = MRI->createVirtualRegister(&AMDGPU::SReg_32RegClass);
  MachineInstr *FlagPHI = BuildMI(*JoinBB, JoinBB->begin(), 
                                   SpillMI->getDebugLoc(),
                                   TII->get(TargetOpcode::PHI), FlagReg);
  
  for (MachineBasicBlock *Pred : JoinBB->predecessors()) {
    FlagPHI->addOperand(MachineOperand::CreateReg(PredToFlagReg[Pred], false));
    FlagPHI->addOperand(MachineOperand::CreateMBB(Pred));
  }
  LIS->InsertMachineInstrInMaps(*FlagPHI);
  
  LLVM_DEBUG(dbgs() << "      Inserted flag PHI: " << *FlagPHI << "\n");
  
  // Step 3 & 4: Create new blocks in order that matches their physical layout
  // Create ReloadBB first, then UseBB_Post, so block numbers match physical order
  // This is required for insertMBBInMaps which asserts prevMBB has lower number
  
  MachineBasicBlock *UseBB_Pre = UseBB;
  
  // Create ReloadBB first (gets lower block number)
  MachineBasicBlock *ReloadBB = MF->CreateMachineBasicBlock(UseBB->getBasicBlock());
  
  // Create UseBB_Post second (gets higher block number)
  MachineBasicBlock *UseBB_Post = MF->CreateMachineBasicBlock(UseBB->getBasicBlock());
  
  // Insert ReloadBB right after UseBB_Pre
  MF->insert(std::next(MachineFunction::iterator(UseBB_Pre)), ReloadBB);
  
  // Insert UseBB_Post right after ReloadBB
  MF->insert(std::next(MachineFunction::iterator(ReloadBB)), UseBB_Post);
  
  // Move instructions from ReloadMI onwards to UseBB_Post
  UseBB_Post->splice(UseBB_Post->begin(), UseBB, ReloadMI, UseBB->end());
  
  // Transfer successors from UseBB_Pre to UseBB_Post
  UseBB_Post->transferSuccessorsAndUpdatePHIs(UseBB_Pre);
  
  LLVM_DEBUG(dbgs() << "      Created ReloadBB and UseBB_Post in correct order\n");
  
  // Move ReloadMI into ReloadBB (it's currently in UseBB_Post)
  ReloadBB->splice(ReloadBB->begin(), UseBB_Post, ReloadMI);
  
  // ReloadBB falls through to UseBB_Post
  ReloadBB->addSuccessor(UseBB_Post);
  
  LLVM_DEBUG(dbgs() << "      Created ReloadBB with reload instruction\n");
  
  // Step 5: Update UseBB_Pre to have conditional branch
  // Layout: UseBB_Pre → ReloadBB → UseBB_Post
  // Branch: if (FlagReg == 0) goto UseBB_Post, else fallthrough to ReloadBB
  // This ensures: FlagReg=1 (came from spill path) → fallthrough to ReloadBB
  //               FlagReg=0 (came from clean path) → branch to UseBB_Post
  
  // Insert conditional branch at end of UseBB_Pre
  // First, compare FlagReg to 0 to set SCC
  MachineInstr *CmpMI = BuildMI(*UseBB_Pre, UseBB_Pre->end(), SpillMI->getDebugLoc(),
                                 TII->get(AMDGPU::S_CMP_EQ_U32))
                            .addReg(FlagReg)
                            .addImm(0);
  LIS->InsertMachineInstrInMaps(*CmpMI);
  
  // Then branch if SCC=1 (flag == 0, clean path) to UseBB_Post
  // Otherwise fallthrough to ReloadBB (flag == 1, spill path)
  MachineInstr *BranchMI = BuildMI(*UseBB_Pre, UseBB_Pre->end(), SpillMI->getDebugLoc(),
                                    TII->get(AMDGPU::S_CBRANCH_SCC1))
                               .addMBB(UseBB_Post);
  LIS->InsertMachineInstrInMaps(*BranchMI);
  
  // Successors: branch to UseBB_Post (clean path), fallthrough to ReloadBB (spill path)
  UseBB_Pre->addSuccessor(UseBB_Post);  // branch target
  UseBB_Pre->addSuccessor(ReloadBB);    // fallthrough
  
  LLVM_DEBUG(dbgs() << "      Inserted conditional branch in UseBB_Pre\n");
  
  // Step 6: Update dominator tree
  // ReloadBB is dominated by UseBB_Pre
  // UseBB_Post is dominated by UseBB_Pre (both paths go through it)
  DT->addNewBlock(UseBB_Post, UseBB_Pre);
  DT->addNewBlock(ReloadBB, UseBB_Pre);
  
  // Register new blocks with SlotIndexes in block number order
  // ReloadBB created first → lower block number
  // UseBB_Post created second → higher block number  
  // Physical order matches number order: UseBB_Pre → ReloadBB → UseBB_Post
  // insertMBBInMaps asserts: mbb->getNumber() == MBBRanges.size()
  Indexes->insertMBBInMaps(ReloadBB);
  Indexes->insertMBBInMaps(UseBB_Post);
  
  LLVM_DEBUG(dbgs() << "      Updated DominatorTree and registered new blocks with SlotIndexes\n");
  
  // Step 7: MachineLaneSSAUpdater will insert value PHI at UseBB_Post
  // when it detects that the reloaded value doesn't dominate all uses
  // The PHI will merge: PHI(original_value from UseBB_Pre, reloaded_value from ReloadBB)
  
  LLVM_DEBUG(dbgs() << "      Conditional reload CFG created successfully\n");
  LLVM_DEBUG(dbgs() << "      MachineLaneSSAUpdater will insert value PHI at UseBB_Post\n");
  
  // Return the reload instruction (still in ReloadBB)
  return ReloadMI;
}

void AMDGPUSSARegisterSpiller::handleUniformReachableUse(
    MachineInstr *SpillMI, MachineInstr *ReloadMI, VRegMaskPair SpilledVMP,
    const SmallVectorImpl<MachineBasicBlock *> &IDFBlocks) {
  
  LLVM_DEBUG(dbgs() << "  Uniform case (no EXEC write) - splitting block\n");
  
  // Split join block, placing reload on spill-path edge only
  [[maybe_unused]] MachineInstr *ReloadInSplitBlock = splitBlockBeforeReload(SpillMI, ReloadMI, SpilledVMP, IDFBlocks);
  
  // MachineLaneSSAUpdater will insert PHI automatically
  // No WWM needed - EXEC is stable
  
  // TODO: Implement cost model for balanced spill decision:
  //  - If avgRP_clean >= 0.8*RPLimit AND len_clean >= 50:
  //    Insert additional spill on clean path after last use
  //  - This avoids redundant memory traffic when clean path also benefits from spilling
}

void AMDGPUSSARegisterSpiller::handleDivergentReachableUse(
    MachineInstr *SpillMI, MachineInstr *ReloadMI, VRegMaskPair SpilledVMP,
    const SmallVectorImpl<MachineBasicBlock *> &IDFBlocks) {
  
  LLVM_DEBUG(dbgs() << "  Divergent case (EXEC write exists) - splitting block + WWM\n");
  
  // Split join block, placing reload on spill-path edge only
  [[maybe_unused]] MachineInstr *ReloadInSplitBlock = splitBlockBeforeReload(SpillMI, ReloadMI, SpilledVMP, IDFBlocks);
  
  // Wrap with WWM for correctness (EXEC drift protection)
  wrapWithWWM(SpillMI, ReloadInSplitBlock);
  
  // MachineLaneSSAUpdater will insert PHI automatically
  
  // TODO: Implement cost model for WWM vs EWF decision:
  //  - Find first EXEC write on path: d_ewf = distance from spill to EXEC write
  //  - Compute f_ewf = d_ewf / NUD (fraction of sink distance lost)
  //  - If f_ewf <= 0.25: Prefer EWF (move reload to EXEC write point)
  //  - Else: Prefer WWM (keep optimal placement, pay SGPR cost)
  //  - If no transient SGPR pair available: Fallback to EWF
}

void AMDGPUSSARegisterSpiller::wrapWithWWM(
    MachineInstr *SpillMI, MachineInstr *ReloadMI) {
  
  // WWM (Whole Wave Mode) wrapping ensures all lanes are spilled/reloaded
  // regardless of current EXEC mask. This prevents EXEC drift correctness issues.
  //
  // Transformation:
  //   Before:
  //     SPILL %vreg (only active lanes per EXEC)
  //     ...
  //     RELOAD %vreg (only active lanes per EXEC)
  //
  //   After:
  //     %save1 = COPY $exec
  //     $exec = S_MOV_B64 -1 (or S_MOV_B32 for wave32)
  //     SPILL %vreg (all lanes)
  //     $exec = COPY %save1
  //     ...
  //     %save2 = COPY $exec
  //     $exec = S_MOV_B64 -1
  //     RELOAD %vreg (all lanes)
  //     $exec = COPY %save2
  
  MachineBasicBlock *SpillBB = SpillMI->getParent();
  MachineBasicBlock *ReloadBB = ReloadMI->getParent();
  
  // Use SIRegisterInfo methods for wave-size agnostic code
  const TargetRegisterClass *ExecRC = TRI->getWaveMaskRegClass();
  MCRegister ExecReg = TRI->getExec();
  
  LLVM_DEBUG(dbgs() << "    Wrapping with WWM, EXEC class: " 
                    << TRI->getRegClassName(ExecRC) << "\n");
  
  // Allocate temporary SGPRs for EXEC save/restore
  // TODO: Implement proper transient SGPR allocation
  // For now, create virtual SGPRs and rely on RA to allocate them
  Register ExecSave1 = MRI->createVirtualRegister(ExecRC);
  Register ExecSave2 = MRI->createVirtualRegister(ExecRC);
  
  // Determine correct MOV opcode based on EXEC register size (wave32 vs wave64)
  unsigned MovOpc = TRI->getRegSizeInBits(ExecReg, *MRI) == 32 
                        ? AMDGPU::S_MOV_B32 
                        : AMDGPU::S_MOV_B64;
  
  // Wrap spill with WWM
  {
    auto InsertPt = SpillMI->getIterator();
    
    // Save EXEC before spill
    MachineInstr *SaveMI = BuildMI(*SpillBB, InsertPt, SpillMI->getDebugLoc(),
                                   TII->get(AMDGPU::COPY), ExecSave1)
                               .addReg(ExecReg);
    LIS->InsertMachineInstrInMaps(*SaveMI);
    
    // Set EXEC to all-ones
    MachineInstr *SetMI = BuildMI(*SpillBB, InsertPt, SpillMI->getDebugLoc(),
                                  TII->get(MovOpc), ExecReg)
                              .addImm(-1);
    LIS->InsertMachineInstrInMaps(*SetMI);
    
    // Spill happens here (SpillMI already inserted)
    
    // Restore EXEC after spill
    auto RestorePt = std::next(SpillMI->getIterator());
    MachineInstr *RestoreMI = BuildMI(*SpillBB, RestorePt, SpillMI->getDebugLoc(),
                                      TII->get(AMDGPU::COPY), ExecReg)
                                  .addReg(ExecSave1);
    LIS->InsertMachineInstrInMaps(*RestoreMI);
    
    LLVM_DEBUG(dbgs() << "      Inserted WWM wrapper around spill\n");
  }
  
  // Wrap reload with WWM
  {
    auto InsertPt = ReloadMI->getIterator();
    
    // Save EXEC before reload
    MachineInstr *SaveMI = BuildMI(*ReloadBB, InsertPt, ReloadMI->getDebugLoc(),
                                   TII->get(AMDGPU::COPY), ExecSave2)
                               .addReg(ExecReg);
    LIS->InsertMachineInstrInMaps(*SaveMI);
    
    // Set EXEC to all-ones
    MachineInstr *SetMI = BuildMI(*ReloadBB, InsertPt, ReloadMI->getDebugLoc(),
                                  TII->get(MovOpc), ExecReg)
                              .addImm(-1);
    LIS->InsertMachineInstrInMaps(*SetMI);
    
    // Reload happens here (ReloadMI already inserted)
    
    // Restore EXEC after reload
    auto RestorePt = std::next(ReloadMI->getIterator());
    MachineInstr *RestoreMI = BuildMI(*ReloadBB, RestorePt, ReloadMI->getDebugLoc(),
                                      TII->get(AMDGPU::COPY), ExecReg)
                                  .addReg(ExecSave2);
    LIS->InsertMachineInstrInMaps(*RestoreMI);
    
    LLVM_DEBUG(dbgs() << "      Inserted WWM wrapper around reload\n");
  }
  
  LLVM_DEBUG(dbgs() << "    WWM wrapping complete\n");
  
  // TODO: Check if transient SGPR pair is available
  // TODO: If not available, fallback to EWF placement instead
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
  bool ChangedSGPR = processFunction(MF, SGPRLimit, /*IsVGPRPass=*/false);

  // Pass 2: VGPR Spilling
  LLVM_DEBUG(dbgs() << "\n=== Pass 2: Processing VGPRs ===\n");
  bool ChangedVGPR = processFunction(MF, VGPRLimit, /*IsVGPRPass=*/true);

  LLVM_DEBUG(dbgs() << "\nAMDGPUSSARegisterSpiller: Completed processing "
                    << MF.getName() << "\n");
  LLVM_DEBUG(dbgs() << "Total spills: " << NumSpills << ", Total reloads: "
                    << NumReloads << "\n");

  // Return true if either pass made modifications
  return ChangedSGPR || ChangedVGPR;
}

// Create function for pass manager
MachineFunctionPass *llvm::createAMDGPUSSARegisterSpillerPass() {
  return new AMDGPUSSARegisterSpiller();
}

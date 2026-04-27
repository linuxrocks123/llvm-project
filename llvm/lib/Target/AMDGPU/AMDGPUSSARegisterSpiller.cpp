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
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "VRegMaskPair.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-ssa-register-spiller"

STATISTIC(NumSpills, "Number of register spills");
STATISTIC(NumReloads, "Number of register reloads");

static cl::opt<bool> EnableVirtualSpillMarkers(
    "amdgpu-ssa-spill-markers",
    cl::desc("Emit SI_VIRTUAL_SPILL_MARKER instructions for SSA spiller tests"),
    cl::Hidden, cl::init(false));

// ============================================================================
static cl::opt<bool> DisableReloadOptimizer(
    "amdgpu-ssa-spill-no-reload-opt",
    cl::desc("Disable reload optimizer in SSA spiller"),
    cl::init(false), cl::Hidden);

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

Printable printVRegMaskPairSet(const VRegMaskPairSet &VMPSet) {
  return Printable([&](raw_ostream &OS) { VMPSet.dump(); });
}

void AMDGPUSSARegisterSpiller::validateFinalRegisterPressure(
    MachineFunction &MF, unsigned RPLimit, bool IsVGPR) {
  
  const char *RegClassName = IsVGPR ? "VGPR" : "SGPR";
  
  LLVM_DEBUG(dbgs() << "\n=== Validating Final Register Pressure ("
                    << RegClassName << ") ===\n");
  
  // Traverse basic blocks same as in processFunction
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  
  for (MachineBasicBlock *MBB : RPOT) {
    if (MBB->empty())
      continue;
    
    // Walk forward through the block
    for (auto I = MBB->begin(), E = MBB->end(); I != E; ++I) {
      MachineInstr &MI = *I;
      
      // Skip spill/reload instructions (same as in processFunction)
      if (isSpillInstr(&MI) || isReloadInstr(&MI))
        continue;
      
      // Reset tracker to compute pressure at this instruction
      RPTracker->reset(MI);
      
      // Get current pressure
      GCNRegPressure CurPressure = RPTracker->getPressure();
      const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
      unsigned CurRP = IsVGPR ? CurPressure.getVGPRNum(ST.hasGFX90AInsts())
                              : CurPressure.getSGPRNum();
      
      if (CurRP > RPLimit) {
        std::string Msg;
        raw_string_ostream OS(Msg);
        OS << "SSA Spiller FINAL RP VALIDATION FAILED!\n";
        OS << "  Register class: " << RegClassName << "\n";
        OS << "  Current RP: " << CurRP << "\n";
        OS << "  RP Limit: " << RPLimit << "\n";
        OS << "  At instruction: " << MI << "\n";
        OS << "  In block: " << printMBBReference(*MBB) << "\n";
        OS << "\nThis indicates the spiller failed to keep RP within limits.\n";
  report_fatal_error(Twine(OS.str()));
      }
    }
  }
  
  LLVM_DEBUG(dbgs() << "✅ Final RP validation passed for " << RegClassName << "\n");
}

bool AMDGPUSSARegisterSpiller::processFunction(MachineFunction &MF,
                                               unsigned RPLimit) {
  LLVM_DEBUG(dbgs() << "processFunction: " << (IsVGPRPass ? "VGPR" : "SGPR")
                    << " pass, limit=" << RPLimit << "\n");

  // Initialize SSA updater (reused throughout the pass, caches IDF computations)
  // FIXME: Clear cache if CFG changes during spilling
  SSAUpdater = std::make_unique<MachineLaneSSAUpdater>(MF, *LIS, *DT, *TRI);

  // Initialize register pressure tracker (reused throughout the pass)
  RPTracker = std::make_unique<GCNUpwardRPTracker>(*LIS);

  // Store RP limits for reload budget checking
  if (IsVGPRPass)
    VGPRLimit = RPLimit;
  else
    SGPRLimit = RPLimit;

  // Track if we made any modifications
  bool Changed = false;

  // Traverse basic blocks in reverse post-order (RPO)
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);

  ReloadedRegs.clear();

  for (MachineBasicBlock *MBB : RPOT) {
    LLVM_DEBUG(dbgs() << "\nProcessing " << printMBBReference(*MBB) << "\n");

    if (MBB->empty())
      continue;

    // Traverse instructions forward (from beginning to end)
    // When we spill at point P, pressure drops from P forward
    //
    // Design Note: We use reset() + forward walk (not recede() + backward walk) because:
    // - Spill insertion at I reduces pressure from I *forward* (down in control flow)
    // - Walking forward with reset(I) naturally sees reduced pressure at I+1 after spilling at I
    // - Walking backward with recede() would detect high RP at I *after* already processing
    //   instructions I+1, I+2, ... that would benefit from the spill (timing mismatch)
    // - reset() cost is O(n) per instruction, acceptable for typical block sizes
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
      RPTracker->reset(MI);

      // Get current register pressure at this point
      GCNRegPressure CurPressure = RPTracker->getPressure();
      
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

        LLVM_DEBUG(dbgs() << "ActiveRegs: " << printVRegMaskPairSet(ActiveRegs) << "\n");
        LLVM_DEBUG(dbgs() << "ReloadedRegs: " << printVRegMaskPairSet(ReloadedRegs) << "\n");
        ActiveRegs.set_subtract(ReloadedRegs);
        LLVM_DEBUG(dbgs() << "ActiveRegs after subtracting ReloadedRegs: "
                          << printVRegMaskPairSet(ActiveRegs) << "\n");

        // CRITICAL: Exclude registers DEFINED by the current instruction!
        // RPTracker.reset(MI) gives us RP AFTER MI executes, which includes
        // registers defined by MI. But we insert spills BEFORE MI, so we
        // cannot spill a register that doesn't exist yet. We must exclude
        // MI.defs().
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
        ActiveRegs.set_subtract(ToRemove);

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
  
  // TEMPORARY: Validate final RP after all spilling is complete
  validateFinalRegisterPressure(MF, RPLimit, IsVGPRPass);
  
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

  // Step 2.5: Loop-aware candidate filtering
  // If spill point is inside a loop, we hoist it to the outermost preheader.
  // Only candidates whose def dominates the effective kill point are valid.
  MachineBasicBlock *EffectiveKillBB = getEffectiveKillBB(&MBB);
  
  if (EffectiveKillBB != &MBB) {
    // Spill point was hoisted - filter candidates
    LLVM_DEBUG(dbgs() << "getVMPsToSpill(): Spill in loop, effective kill at "
                      << printMBBReference(*EffectiveKillBB) << "\n");
    
    // Rebuild Active with only valid candidates (preserving NUD order)
    SmallVector<VRegMaskPair> ValidCandidates;
    for (const auto &VMP : Active) {
      MachineInstr *DefMI = MRI->getVRegDef(VMP.getVReg());
      if (!DefMI)
        continue;
      
      MachineBasicBlock *DefBB = DefMI->getParent();
      // Def must dominate the effective (hoisted) kill point
      if (DT->dominates(DefBB, EffectiveKillBB)) {
        ValidCandidates.push_back(VMP);
        LLVM_DEBUG({
          StringRef Name = MRI->getVRegName(VMP.getVReg());
          dbgs() << "  Valid candidate: ";
          if (!Name.empty())
            dbgs() << "%" << Name;
          else
            dbgs() << printReg(VMP.getVReg(), TRI);
          dbgs() << " (def dominates effective kill)\n";
        });
      } else {
        LLVM_DEBUG({
          StringRef Name = MRI->getVRegName(VMP.getVReg());
          dbgs() << "  Filtered out: ";
          if (!Name.empty())
            dbgs() << "%" << Name;
          else
            dbgs() << printReg(VMP.getVReg(), TRI);
          dbgs() << " (def in loop, doesn't dominate effective kill)\n";
        });
      }
    }
    
    if (ValidCandidates.empty()) {
      LLVM_DEBUG(dbgs() << "getVMPsToSpill(): No valid candidates after loop filter!\n");
      // TODO: Fallback - pick best invalid candidate and use loop exit sinking
      return ToSpill;
    }
    
    // Rebuild Active from valid candidates (preserves NUD order)
    Active.clear();
    for (const auto &VMP : ValidCandidates)
      Active.insert(VMP);
  }

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

void AMDGPUSSARegisterSpiller::insertVirtualSpillMarker(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
    VRegMaskPair VMP) {
  // Avoid dropping a marker immediately after the actual spill store
  // for the same VReg/Lane mask.
  MachineInstr *PrevMI = nullptr;
  if (I == MBB.end()) {
    if (!MBB.empty()) {
      auto PrevIt = MBB.end();
      --PrevIt;
      PrevMI = &*PrevIt;
    }
  } else if (I != MBB.begin()) {
    auto PrevIt = I;
    --PrevIt;
    PrevMI = &*PrevIt;
  }

  if (!PrevMI || !isSpillInstr(PrevMI) || !usesSpilledVMP(PrevMI, VMP)) {
    // Insert MIR-visible marker so tests can assert the virtual spill point.
    DebugLoc SpillDL =
        I == MBB.end() ? DebugLoc() : I->getDebugLoc();
    MachineInstr *MarkerMI = BuildMI(MBB, I, SpillDL,
                                     TII->get(AMDGPU::SI_VIRTUAL_SPILL_MARKER))
                                 .addImm(VMP.getVReg().virtRegIndex())
                                 .addImm(VMP.getLaneMask().getAsInteger());
    LIS->InsertMachineInstrInMaps(*MarkerMI);
  } else {
    LLVM_DEBUG(
        dbgs()
        << "Skipping virtual spill marker (adjacent real spill of same VMP)\n");
  }
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

  // TODO: this message is not correct. spillAndReload will spill as much as CurRP - RPLimit,
  // but here we print a total number of VMPs available for spill.
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

    // Step 2a: Store register at definition point (when EXEC is full)
    // This avoids EXEC drift issues by ensuring all lanes are stored before
    // any divergent control flow can modify EXEC
    MachineInstr *DefStoreMI = spillAtDefinition(VMP);
    assert(DefStoreMI && "Virtual register must have a definition in SSA form");
    
    // Step 2b: Set virtual "spill point" at the high-pressure point
    // This is where RP exceeded, but we don't prune the LiveInterval here.
    // The LiveInterval will be shrunk later by shrinkToUses() after all reloads are placed.
    MachineBasicBlock::iterator SpillPos = I.getReverse();
    
    // Get the effective kill block (hoisted out of loops if needed)
    MachineBasicBlock *EffectiveKillBB = getEffectiveKillBB(&MBB);
    
    SlotIndex KillIdx;
    MachineBasicBlock::iterator MarkerPos;
    MachineBasicBlock *MarkerBB = EffectiveKillBB;
    
    if (EffectiveKillBB != &MBB) {
      // Spill hoisted to preheader - kill at preheader end
      KillIdx = Indexes->getMBBEndIdx(EffectiveKillBB).getPrevSlot();
      MarkerPos = EffectiveKillBB->getFirstTerminator();
    } else if (SpillPos == MBB.end()) {
      KillIdx = Indexes->getMBBEndIdx(&MBB).getPrevSlot();
      MarkerPos = MBB.end();
    } else {
      KillIdx = Indexes->getInstructionIndex(*SpillPos).getRegSlot();
      MarkerPos = SpillPos;
    }
    
    // Virtual spill marker at effective kill point
    if (EnableVirtualSpillMarkers) {
      if (I->isPHI() && EffectiveKillBB == &MBB) {
        // PHI case only applies when not hoisted
        LLVM_DEBUG(dbgs() << "Virtual spill marker for PHI goes to predecessors\n");
        for (auto *Pred : MBB.predecessors()) {
          insertVirtualSpillMarker(*Pred, Pred->getFirstTerminator(), VMP);
        }
      } else {
        insertVirtualSpillMarker(*MarkerBB, MarkerPos, VMP);
      }
    }
    
    LLVM_DEBUG({
      dbgs() << "spillAndReload(): Virtual spill point (KillIdx): " << KillIdx << "\n";
    });
    
    // Step 2c: Get stack slot for reload phase
    int FI = assignVirt2StackSlot(VMP);
    
    // Step 2d: Build SpillInfo with dom-groups and emit reloads
    SpillInfo Info;
    Info.SpilledVMP = VMP;
    Info.KillIdx = KillIdx;
    Info.FrameIndex = FI;
    buildDomGroupsForSpill(Info);
    emitReloadsAndRepairSSA(Info);
  }

  LLVM_DEBUG(dbgs() << "spillAndReload(): Completed, spilled " 
                    << ToSpill.size() << " VMP(s)\n");
  LLVM_DEBUG(dbgs() << "===================================\n\n");

  return true;
}

// ============================================================================
// Reload Optimizer
// ============================================================================

// TODO: Investigate profitability/possibility of early return when RP > Limit.
// Callers only care whether RP exceeds the limit, not by how much.
// Optimization: if we find RP > Limit at any point, return early and cache
// that value - no need to compute the actual maximum.
unsigned AMDGPUSSARegisterSpiller::getMaxRPForBlock(MachineBasicBlock *MBB) {
  auto It = MaxRPCache.find(MBB);
  if (It != MaxRPCache.end())
    return It->second;
  
  // Compute max RP by tracking backwards through the block
  GCNUpwardRPTracker Tracker(*LIS);
  Tracker.reset(*MBB);
  
  const GCNSubtarget &ST = MBB->getParent()->getSubtarget<GCNSubtarget>();
  
  // Include initial pressure (live-out at block end)
  GCNRegPressure InitPressure = Tracker.getPressure();
  unsigned MaxRP = IsVGPRPass ? InitPressure.getVGPRNum(ST.hasGFX90AInsts())
                              : InitPressure.getSGPRNum();
  
  for (MachineInstr &MI : reverse(*MBB)) {
    if (MI.isDebugInstr())
      continue;
    Tracker.recede(MI);
    GCNRegPressure Pressure = Tracker.getPressure();
    unsigned CurRP = IsVGPRPass ? Pressure.getVGPRNum(ST.hasGFX90AInsts())
                                : Pressure.getSGPRNum();
    MaxRP = std::max(MaxRP, CurRP);
  }
  
  MaxRPCache[MBB] = MaxRP;
  return MaxRP;
}

unsigned AMDGPUSSARegisterSpiller::getMaxRPInBlockDownTo(MachineBasicBlock *MBB,
                                                        MachineInstr *StopMI) {
  if (!StopMI || StopMI->getParent() != MBB)
    return getMaxRPForBlock(MBB);
  
  // Compute max RP from block start up to (not including) StopMI
  GCNUpwardRPTracker Tracker(*LIS);
  
  // Start from StopMI and track backwards to block start
  Tracker.reset(*StopMI);
  
  unsigned MaxRP = 0;
  const GCNSubtarget &ST = MBB->getParent()->getSubtarget<GCNSubtarget>();
  
  for (auto It = StopMI->getReverseIterator(); It != MBB->rend(); ++It) {
    MachineInstr &MI = *It;
    if (MI.isDebugInstr())
      continue;
    Tracker.recede(MI);
    GCNRegPressure Pressure = Tracker.getPressure();
    unsigned CurRP = IsVGPRPass ? Pressure.getVGPRNum(ST.hasGFX90AInsts())
                                : Pressure.getSGPRNum();
    MaxRP = std::max(MaxRP, CurRP);
  }
  
  return MaxRP;
}

bool AMDGPUSSARegisterSpiller::canHoistReloadTo(MachineBasicBlock *NCD,
                                                 MachineInstr *InsertPoint,
                                                 unsigned RPLimit,
                                                 Register SpilledReg) {
  // Spilled register is already counted as live, so MaxRP > RPLimit means
  // no room for reload (no +1 needed).
  
  // Check RP in NCD block only if reload is placed inside NCD (InsertPoint set)
  // If InsertPoint is nullptr, reload goes at NCD end - skip NCD RP check,
  // walkPathsToUses will check paths from NCD to uses.
  if (InsertPoint) {
    unsigned NCDRP = getMaxRPInBlockDownTo(NCD, InsertPoint);
    if (NCDRP > RPLimit)
      return false;
  }
  
  auto IsHighRP = [&](MachineBasicBlock *BB, MachineInstr *UseMI) -> bool {
    unsigned CurRP = UseMI ? getMaxRPInBlockDownTo(BB, UseMI)
                           : getMaxRPForBlock(BB);
    return CurRP > RPLimit;
  };
  
  return walkPathsToUses(NCD, SpilledReg, IsHighRP);
}

SmallVector<std::pair<MachineBasicBlock *, MachineInstr *>, 4>
AMDGPUSSARegisterSpiller::optimizeReloadPlacing(
    const SmallVectorImpl<MachineInstr *> &GroupHeads, unsigned RPLimit,
    Register SpilledReg) {
  
  using ResultPair = std::pair<MachineBasicBlock *, MachineInstr *>;
  SmallVector<ResultPair, 4> Result;
  
  if (GroupHeads.empty())
    return Result;
  
  // Single group - no optimization possible
  if (GroupHeads.size() == 1) {
    MachineInstr *Head = GroupHeads[0];
    Result.push_back({Head->getParent(), Head});
    return Result;
  }
  
  LLVM_DEBUG(dbgs() << "\n=== Reload Optimizer: " << GroupHeads.size() 
                    << " group heads ===\n");
  
  unsigned N = GroupHeads.size();
  
  // Build NCD matrix and InsertPoint matrix
  // NCDMatrix[i][j] = NCD of GroupHeads[i] and GroupHeads[j]
  // InsertPointMatrix[i][j] = reload insertion point in NCD (earliest of the two if both in NCD)
  SmallVector<SmallVector<MachineBasicBlock *, 4>, 4> NCDMatrix(N);
  SmallVector<SmallVector<MachineInstr *, 4>, 4> InsertPointMatrix(N);
  
  for (unsigned i = 0; i < N; ++i) {
    NCDMatrix[i].resize(N, nullptr);
    InsertPointMatrix[i].resize(N, nullptr);
  }
  
  // Fill matrices
  for (unsigned i = 0; i < N; ++i) {
    MachineBasicBlock *BBi = GroupHeads[i]->getParent();
    NCDMatrix[i][i] = BBi;
    InsertPointMatrix[i][i] = GroupHeads[i];
    
    for (unsigned j = i + 1; j < N; ++j) {
      MachineBasicBlock *BBj = GroupHeads[j]->getParent();
      MachineBasicBlock *NCD = DT->findNearestCommonDominator(BBi, BBj);
      
      if (!NCD) {
        // No common dominator - can't merge
        NCDMatrix[i][j] = NCDMatrix[j][i] = nullptr;
        continue;
      }
      
      NCDMatrix[i][j] = NCDMatrix[j][i] = NCD;
      
      // Determine InsertPoint: earliest instruction if both uses are in NCD
      MachineInstr *InsertPoint = nullptr;
      if (NCD == BBi && NCD == BBj) {
        // Both in same block - use the earlier one
        for (MachineInstr &MI : *NCD) {
          if (&MI == GroupHeads[i] || &MI == GroupHeads[j]) {
            InsertPoint = &MI;
            break;
          }
        }
      } else if (NCD == BBi) {
        InsertPoint = GroupHeads[i];
      } else if (NCD == BBj) {
        InsertPoint = GroupHeads[j];
      }
      // else: NCD is a dominator of both, InsertPoint = nullptr (end of block)
      
      InsertPointMatrix[i][j] = InsertPointMatrix[j][i] = InsertPoint;
    }
  }
  
  // Build adjacency: edge[i][j] = true if we can hoist both i and j to their NCD
  SmallVector<SmallVector<bool, 4>, 4> CanMerge(N);
  SmallVector<unsigned, 4> EdgeCount(N, 0);
  
  for (unsigned i = 0; i < N; ++i)
    CanMerge[i].resize(N, false);
  
  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = i + 1; j < N; ++j) {
      MachineBasicBlock *NCD = NCDMatrix[i][j];
      if (!NCD)
        continue;
      
      MachineInstr *InsertPoint = InsertPointMatrix[i][j];
      
      // Check NCD RP only if reload is placed inside NCD (use exists in NCD)
      // If InsertPoint is nullptr, reload goes at NCD end - skip NCD RP check
      if (InsertPoint) {
        unsigned NCDRP = getMaxRPInBlockDownTo(NCD, InsertPoint);
        if (NCDRP > RPLimit)
          continue;
      }
      
      // Single-walk BFS from NCD, checking RP along the way
      // Liveness scoping ensures we only check relevant blocks
      auto IsHighRP = [&](MachineBasicBlock *BB, MachineInstr *UseMI) -> bool {
        unsigned CurRP = UseMI ? getMaxRPInBlockDownTo(BB, UseMI)
                               : getMaxRPForBlock(BB);
        return CurRP > RPLimit;
      };
      
      if (walkPathsToUses(NCD, SpilledReg, IsHighRP)) {
        CanMerge[i][j] = CanMerge[j][i] = true;
        EdgeCount[i]++;
        EdgeCount[j]++;
      }
    }
  }
  
  // Greedy clique extraction
  SmallVector<int, 4> Assignment(N, -1); // -1 = unassigned
  int NextCliqueID = 0;
  
  while (true) {
    // Find unassigned node with most edges (heuristic for larger cliques)
    int BestSeed = -1;
    unsigned BestEdges = 0;
    for (unsigned i = 0; i < N; ++i) {
      if (Assignment[i] == -1 && EdgeCount[i] >= BestEdges) {
        BestSeed = i;
        BestEdges = EdgeCount[i];
      }
    }
    
    if (BestSeed == -1)
      break; // All assigned
    
    // Start clique with this seed
    SmallVector<unsigned, 4> Clique;
    Clique.push_back(BestSeed);
    Assignment[BestSeed] = NextCliqueID;
    
    // Try to extend clique with other unassigned nodes
    for (unsigned i = 0; i < N; ++i) {
      if (Assignment[i] != -1)
        continue;
      
      // Check if i can join the clique (connected to all current members)
      bool CanJoin = true;
      for (unsigned Member : Clique) {
        if (!CanMerge[i][Member]) {
          CanJoin = false;
          break;
        }
      }
      
      if (CanJoin) {
        Clique.push_back(i);
        Assignment[i] = NextCliqueID;
      }
    }
    
    // Compute NCD for this clique
    MachineBasicBlock *CliqueNCD = GroupHeads[Clique[0]]->getParent();
    MachineInstr *CliqueInsertPoint = GroupHeads[Clique[0]];
    
    for (unsigned i = 1; i < Clique.size(); ++i) {
      MachineBasicBlock *BB = GroupHeads[Clique[i]]->getParent();
      CliqueNCD = DT->findNearestCommonDominator(CliqueNCD, BB);
      
      // Update InsertPoint if use is in NCD
      if (CliqueNCD == BB) {
        MachineInstr *Head = GroupHeads[Clique[i]];
        if (!CliqueInsertPoint || CliqueInsertPoint->getParent() != CliqueNCD) {
          CliqueInsertPoint = Head;
        } else {
          // Both in same block - find earlier
          for (MachineInstr &MI : *CliqueNCD) {
            if (&MI == Head || &MI == CliqueInsertPoint) {
              CliqueInsertPoint = &MI;
              break;
            }
          }
        }
      } else if (CliqueInsertPoint && CliqueInsertPoint->getParent() != CliqueNCD) {
        // NCD changed and InsertPoint is no longer in NCD
        CliqueInsertPoint = nullptr;
      }
    }
    
    // Add result: one reload for this clique
    if (CliqueInsertPoint && CliqueInsertPoint->getParent() == CliqueNCD) {
      Result.push_back({CliqueNCD, CliqueInsertPoint});
    } else {
      // Insert at end of NCD (before terminator)
      Result.push_back({CliqueNCD, nullptr});
    }
    
    LLVM_DEBUG({
      dbgs() << "  Clique " << NextCliqueID << ": {";
      for (unsigned i = 0; i < Clique.size(); ++i) {
        if (i > 0) dbgs() << ", ";
        dbgs() << Clique[i];
      }
      dbgs() << "} -> " << printMBBReference(*CliqueNCD);
      if (CliqueInsertPoint)
        dbgs() << " before " << *CliqueInsertPoint;
      dbgs() << "\n";
    });
    
    NextCliqueID++;
  }
  
  LLVM_DEBUG(dbgs() << "  Reduced " << N << " groups to " << Result.size() 
                    << " reload points\n");
  
  return Result;
}

void AMDGPUSSARegisterSpiller::fixPathologicalPHIs(VRegMaskPair SpilledVMP,
                                                    int FrameIndex,
                                                    MachineInstr *KillMI) {
  Register SpilledReg = SpilledVMP.getVReg();
  
  // Find PHI instructions that still use the spilled register
  // AND are dominated by the spill point
  SmallVector<MachineInstr *, 4> PathologicalPHIs;
  
  for (MachineOperand &MO : MRI->use_operands(SpilledReg)) {
    MachineInstr *UseMI = MO.getParent();
    if (!UseMI->isPHI())
      continue;
    
    // Only PHIs dominated by spill point are pathological
    // PHIs NOT dominated have a clean path (bypass) and should keep original value
    if (!DT->dominates(KillMI, UseMI))
      continue;
    
    // Check if this PHI operand overlaps with spilled lanes
    VRegMaskPair UseVMP(MO, TRI, MRI);
    if (!UseVMP.overlaps(SpilledVMP))
      continue;
    
    PathologicalPHIs.push_back(UseMI);
  }
  
  if (PathologicalPHIs.empty())
    return;
  
  LLVM_DEBUG(dbgs() << "\n=== Fixing " << PathologicalPHIs.size() 
                    << " pathological PHI(s) ===\n");
  
  for (MachineInstr *PHI : PathologicalPHIs) {
    // Get PHI destination - we'll keep this register to avoid rewriting uses
    Register PHIDest = PHI->getOperand(0).getReg();
    MachineBasicBlock *PHIBB = PHI->getParent();
    
    LLVM_DEBUG(dbgs() << "  Replacing PHI: " << *PHI);
    
    // Get the register class from the PHI destination
    const TargetRegisterClass *RC = MRI->getRegClass(PHIDest);
    
    // Insert reload after all PHIs (can't insert non-PHI before PHI)
    MachineBasicBlock::iterator InsertPos = PHIBB->getFirstNonPHI();
    TII->loadRegFromStackSlot(*PHIBB, InsertPos, PHIDest, FrameIndex, RC,
                              TRI, Register());
    MachineInstr *ReloadMI = &*std::prev(InsertPos);
    
    // Add to slot indexes (LiveIntervals recomputed at end via shrinkToUses)
    Indexes->insertMachineInstrInMaps(*ReloadMI);
    
    // Remove the PHI
    Indexes->removeMachineInstrFromMaps(*PHI);
    PHI->eraseFromParent();
    
    ++NumReloads;
    
    LLVM_DEBUG(dbgs() << "    Inserted reload: " << *ReloadMI);
  }
}

// ============================================================================
// Loop-Aware Spilling Helpers
// ============================================================================

bool AMDGPUSSARegisterSpiller::hasDefInLoop(VRegMaskPair VMP) const {
  MachineInstr *DefMI = MRI->getVRegDef(VMP.getVReg());
  if (!DefMI)
    return false;
  return MLI->getLoopFor(DefMI->getParent()) != nullptr;
}

bool AMDGPUSSARegisterSpiller::hasUseInLoop(VRegMaskPair VMP) const {
  Register Reg = VMP.getVReg();
  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(Reg)) {
    // Skip spill instructions
    if (isSpillInstr(&UseMI))
      continue;
    
    // Check if use operand overlaps with VMP
    MachineOperand *UseOp = UseMI.findRegisterUseOperand(Reg, TRI, false);
    if (!UseOp)
      continue;
    VRegMaskPair UseVMP(*UseOp, TRI, MRI);
    if (!UseVMP.overlaps(VMP))
      continue;
    
    if (MLI->getLoopFor(UseMI.getParent()))
      return true;
  }
  return false;
}

MachineBasicBlock *AMDGPUSSARegisterSpiller::getLoopExitDominatingSpill(
    MachineLoop *Loop, MachineBasicBlock *SpillBB) const {
  SmallVector<MachineBasicBlock *, 4> ExitBlocks;
  Loop->getExitBlocks(ExitBlocks);
  
  for (MachineBasicBlock *Exit : ExitBlocks) {
    if (DT->dominates(Exit, SpillBB))
      return Exit;
  }
  return nullptr;
}

MachineBasicBlock *AMDGPUSSARegisterSpiller::getEffectiveKillBB(
    MachineBasicBlock *SpillBB) const {
  // Find outermost loop containing spill point
  MachineLoop *Loop = MLI->getLoopFor(SpillBB);
  if (!Loop)
    return SpillBB;  // Not in any loop
  
  // Walk up to outermost loop
  while (MachineLoop *Parent = Loop->getParentLoop())
    Loop = Parent;
  
  // Get outermost loop's preheader
  MachineBasicBlock *Preheader = Loop->getLoopPreheader();
  if (Preheader) {
    LLVM_DEBUG(dbgs() << "  Hoisting spill point from " << printMBBReference(*SpillBB)
                      << " to preheader " << printMBBReference(*Preheader) << "\n");
    return Preheader;
  }
  
  // Irreducible loop - can't hoist
  LLVM_DEBUG(dbgs() << "  Warning: No preheader for loop containing spill point\n");
  return SpillBB;
}

std::pair<MachineBasicBlock *, MachineInstr *>
AMDGPUSSARegisterSpiller::adjustReloadForLoop(MachineBasicBlock *ReloadBB,
                                               MachineInstr *InsertBeforeMI,
                                               MachineBasicBlock *KillBB,
                                               Register SpilledReg) {
  MachineLoop *ReloadLoop = MLI->getLoopFor(ReloadBB);
  if (ReloadLoop && !ReloadLoop->contains(KillBB)) {
    // Use in loop, spill outside - consider hoisting reload to preheader
    MachineBasicBlock *Preheader = ReloadLoop->getLoopPreheader();
    if (Preheader) {
      unsigned RPLimit = IsVGPRPass ? VGPRLimit : SGPRLimit;
      MachineInstr *InsertPoint = nullptr;
      auto TermIt = Preheader->getFirstTerminator();
      if (TermIt != Preheader->end())
        InsertPoint = &*TermIt;
      
      bool CanHoist = canHoistReloadTo(Preheader, InsertPoint, RPLimit, SpilledReg);

      if (!CanHoist) {
        LLVM_DEBUG(dbgs() << "  Cannot hoist reload to preheader: "
                          << "RP exceeds limit on path, keeping reload inside loop\n");
        return {ReloadBB, InsertBeforeMI};  // Don't hoist - accept reload in loop
      }
      
      LLVM_DEBUG(dbgs() << "  Hoisting reload from " << printMBBReference(*ReloadBB)
                        << " to preheader " << printMBBReference(*Preheader) << "\n");
      return {Preheader, nullptr};  // Insert at end of preheader
    }
  }
  return {ReloadBB, InsertBeforeMI};
}

// ===========================================================================
// IDF-First PHI Insertion Strategy
// ===========================================================================

void AMDGPUSSARegisterSpiller::buildDomGroupsForSpill(SpillInfo &Info) {
  Register SpilledReg = Info.SpilledVMP.getVReg();
  MachineInstr *KillMI = Indexes->getInstructionFromIndex(Info.KillIdx);
  
  LLVM_DEBUG(dbgs() << "buildDomGroupsForSpill for "
                    << printReg(SpilledReg, TRI) << "\n");
  
  // Collect all uses into a vector for sorting
  SmallVector<MachineInstr *, 8> AllUses;
  for (MachineInstr &UseMI : MRI->use_nodbg_instructions(SpilledReg)) {
    if (isSpillInstr(&UseMI) || UseMI.isPHI())
      continue;
    
    MachineOperand *UseOp = UseMI.findRegisterUseOperand(SpilledReg, TRI, /*isKill=*/false);
    if (!UseOp)
      continue;
    
    VRegMaskPair UseVMP(*UseOp, TRI, MRI);
    if (!UseVMP.overlaps(Info.SpilledVMP))
      continue;
    
    // Only consider uses reachable from KillMI
    if (!DT->dominates(KillMI, &UseMI) &&
        !SSAUpdater->isUseReachableFromDef(KillMI, &UseMI, SpilledReg,
                                            Info.SpilledVMP.getLaneMask()))
      continue;
    
    AllUses.push_back(&UseMI);
  }
  
  // Sort uses by dominance order (dominating first)
  llvm::sort(AllUses, [this](MachineInstr *A, MachineInstr *B) {
    if (DT->dominates(A, B))
      return true;
    if (DT->dominates(B, A))
      return false;
    // For unrelated uses, use slot index as tiebreaker
    return Indexes->getInstructionIndex(*A) < Indexes->getInstructionIndex(*B);
  });
  
  // Build groups: for each use, either merge into existing group or create new
  for (MachineInstr *UseMI : AllUses) {
    bool Merged = false;
    for (DomGroup &G : Info.DomGroups) {
      if (DT->dominates(G.getHead(), UseMI)) {
        G.addDominatedUse(UseMI);
        Merged = true;
        break;
      }
      if (DT->dominates(UseMI, G.getHead())) {
        G.promoteHead(UseMI);
        Merged = true;
        break;
      }
    }
    if (!Merged) {
      Info.DomGroups.emplace_back(UseMI);
    }
  }
  
  LLVM_DEBUG(dbgs() << "  Built " << Info.DomGroups.size() << " dom-groups\n");
}

void AMDGPUSSARegisterSpiller::sortByDominanceOrder(
    SmallVectorImpl<MachineBasicBlock *> &Blocks) {
  llvm::sort(Blocks, [this](MachineBasicBlock *A, MachineBasicBlock *B) {
    if (DT->dominates(A, B))
      return true;
    if (DT->dominates(B, A))
      return false;
    return A->getNumber() < B->getNumber();
  });
}

MachineBasicBlock *AMDGPUSSARegisterSpiller::findClosestDominatingPIDF(
    MachineInstr *UseMI,
    const SmallVectorImpl<MachineBasicBlock *> &PIdfBlocks) {
  MachineBasicBlock *UseBB = UseMI->getParent();
  MachineBasicBlock *Closest = nullptr;
  
  for (MachineBasicBlock *PIdf : PIdfBlocks) {
    if (DT->dominates(PIdf, UseBB)) {
      if (!Closest || DT->dominates(Closest, PIdf)) {
        Closest = PIdf;
      }
    }
  }
  return Closest;
}

std::pair<Register, MachineInstr *>
AMDGPUSSARegisterSpiller::getOrCreateReloadInBlock(
    MachineBasicBlock *BB, VRegMaskPair SpilledVMP, MachineInstr *InsertBefore) {
  auto Key = std::make_pair(BB, SpilledVMP.getVReg());
  
  // Only use cache for block-end reloads (InsertBefore == nullptr)
  if (!InsertBefore) {
    auto It = BlockReloadCache.find(Key);
    if (It != BlockReloadCache.end()) {
      LLVM_DEBUG(dbgs() << "    Reusing cached reload in " << printMBBReference(*BB)
                        << ": " << printReg(It->second, TRI) << "\n");
      return {It->second, nullptr};  // Cached - no new instruction
    }
  }
  
  // Create a fresh VReg for this reload (no automatic SSA repair)
  // RC is the class for the spilled lanes (narrower if partial spill)
  const TargetRegisterClass *RC = SpilledVMP.getRegClass(MRI, TRI);
  Register NewVReg = MRI->createVirtualRegister(RC);
  
  // Determine insertion point: before specified instruction or at block end
  auto InsertIt = InsertBefore ? InsertBefore->getIterator() 
                               : BB->getFirstTerminator();
  int FI = assignVirt2StackSlot(SpilledVMP);
  
  // NewVReg is full register of RC - NO subreg index needed
  TII->loadRegFromStackSlot(*BB, InsertIt, NewVReg, FI, RC, TRI,
                            Register(), MachineInstr::NoFlags, /*SubIdx=*/0);
  
  // Get the reload instruction and add to slot indexes
  MachineInstr *ReloadMI = &*std::prev(InsertIt);
  LIS->InsertMachineInstrInMaps(*ReloadMI);
  
  // NOTE: Do NOT compute live interval here - uses haven't been rewritten yet.
  // Live interval will be computed lazily by getInterval() after use rewriting.
  
  // Track reloaded register to prevent re-spilling
  LaneBitmask Mask = SpilledVMP.getLaneMask();
  ReloadedRegs.insert(VRegMaskPair(NewVReg, Mask));
  
  // Cache only block-end reloads
  if (!InsertBefore) {
    BlockReloadCache[Key] = NewVReg;
  }
  
  LLVM_DEBUG(dbgs() << "    Created reload in " << printMBBReference(*BB)
                    << (InsertBefore ? " before use" : " at block end")
                    << ": " << printReg(NewVReg, TRI) << "\n");
  ++NumReloads;
  return {NewVReg, ReloadMI};  // New reload created
}

bool AMDGPUSSARegisterSpiller::insertReloadForUse(
    MachineInstr *UseMI, VRegMaskPair SpilledVMP, MachineBasicBlock *KillBB) {
  Register SpilledReg = SpilledVMP.getVReg();
  LaneBitmask SpilledMask = SpilledVMP.getLaneMask();
  unsigned RPLimit = IsVGPRPass ? VGPRLimit : SGPRLimit;
  
  if (UseMI->isPHI()) {
    // PHI use: reload must be in predecessor block(s) that provide the spilled reg
    bool InsertedAny = false;
    for (unsigned i = 1; i < UseMI->getNumOperands(); i += 2) {
      MachineOperand &ValOp = UseMI->getOperand(i);
      MachineOperand &BBOp = UseMI->getOperand(i + 1);
      if (!ValOp.isReg() || ValOp.getReg() != SpilledReg)
        continue;
      
      // Check if this PHI operand's lanes overlap with spilled lanes
      LaneBitmask UseMask = VRegMaskPair(ValOp, TRI, MRI).getLaneMask();
      if ((UseMask & SpilledMask).none())
        continue;
      
      MachineBasicBlock *PredBB = BBOp.getMBB();
      
      // Check RP in predecessor
      unsigned PredRP = getMaxRPForBlock(PredBB);
      if (PredRP > RPLimit) {
        LLVM_DEBUG(dbgs() << "    WARNING: Predecessor " << printMBBReference(*PredBB)
                          << " has RP=" << PredRP << " > limit=" << RPLimit
                          << ", but must insert reload for PHI use\n");
      }
      
      auto [ReloadReg, ReloadMI] = getOrCreateReloadInBlock(PredBB, SpilledVMP, nullptr);
      if (ReloadMI) {
        // New reload created - use full SSA repair with IDF PHI insertion
        SSAUpdater->repairSSAForReload(ReloadReg, SpilledReg, SpilledMask,
                                       ReloadMI->getParent());
      } else {
        // Cached reload - just rewrite dominated uses
        SSAUpdater->rewriteDominatedUses(SpilledReg, ReloadReg, SpilledMask);
      }
      InsertedAny = true;
      LLVM_DEBUG(dbgs() << "    PHI use: reload in " << printMBBReference(*PredBB) << "\n");
    }
    return InsertedAny;
  }
  
  // Non-PHI use: insert before use with loop adjustment
  auto Adjusted = adjustReloadForLoop(UseMI->getParent(), UseMI, KillBB, SpilledReg);
  MachineInstr *InsertBeforeUse = (Adjusted.first == UseMI->getParent()) ? UseMI : nullptr;
  auto [ReloadReg, ReloadMI] = getOrCreateReloadInBlock(Adjusted.first, SpilledVMP, InsertBeforeUse);
  if (ReloadMI) {
    // New reload created - use full SSA repair with IDF PHI insertion
    SSAUpdater->repairSSAForReload(ReloadReg, SpilledReg, SpilledMask,
                                   ReloadMI->getParent());
  } else {
    // Cached reload - just rewrite dominated uses
    SSAUpdater->rewriteDominatedUses(SpilledReg, ReloadReg, SpilledMask);
  }
  return true;
}

MachineInstr *AMDGPUSSARegisterSpiller::emitReloadToReg(
    MachineBasicBlock::iterator InsertBefore, VRegMaskPair VMP, Register TargetReg) {
  // For now, use the standard reload mechanism - TargetReg is unused
  // TODO: Support emitting reload to a specific register if needed
  (void)TargetReg;
  return emitReload(InsertBefore, VMP);
}

MachineInstr *AMDGPUSSARegisterSpiller::processPIdfBlock(
    MachineBasicBlock *PIdfBB,
    VRegMaskPair SpilledVMP,
    MachineBasicBlock *KillBB,
    const SmallVectorImpl<DomGroup *> &Groups,
    unsigned RPLimit) {
  
  Register SpilledReg = SpilledVMP.getVReg();
  LaneBitmask SpilledMask = SpilledVMP.getLaneMask();
  
  LLVM_DEBUG(dbgs() << "\nprocessPIdfBlock: " << printMBBReference(*PIdfBB)
                    << " with " << Groups.size() << " groups\n");
  
  SmallVector<MachineInstr *, 4> Heads;
  for (DomGroup *G : Groups) {
    Heads.push_back(G->getHead());
  }
  
  SmallPtrSet<MachineInstr *, 4> HighRPHeads;
  
  auto RPCheckFn = [&](MachineBasicBlock *BB, MachineInstr *UseMI) -> bool {
    if (std::find(Heads.begin(), Heads.end(), UseMI) != Heads.end()) {
      unsigned MaxRP = getMaxRPInBlockDownTo(BB, UseMI);
      if (MaxRP > RPLimit) {
        LLVM_DEBUG(dbgs() << "    High RP on path to " << *UseMI);
        HighRPHeads.insert(UseMI);
      }
      return false; // Don't stop - continue to collect all heads
    }
    unsigned MaxRP = getMaxRPForBlock(BB);
    return MaxRP > RPLimit;
  };
  
  // Walk all paths, don't stop early - collect all high-RP heads
  walkPathsToUses(PIdfBB, SpilledReg, RPCheckFn, /*stopOnBad=*/false);
  
  SmallVector<DomGroup *, 4> PhiOKGroups;
  SmallVector<DomGroup *, 4> ReloadAtUseGroups;
  
  for (DomGroup *G : Groups) {
    if (HighRPHeads.count(G->getHead())) {
      ReloadAtUseGroups.push_back(G);
    } else {
      PhiOKGroups.push_back(G);
    }
  }
  
  LLVM_DEBUG(dbgs() << "  PhiOK groups: " << PhiOKGroups.size()
                    << ", ReloadAtUse groups: " << ReloadAtUseGroups.size() << "\n");
  
  if (PhiOKGroups.empty()) {
    LLVM_DEBUG(dbgs() << "  All groups need reload-at-use, no PHI\n");
    for (DomGroup *G : Groups) {
      insertReloadForUse(G->getHead(), SpilledVMP, KillBB);
    }
    return nullptr;
  }
  
  // Check if any spill-path predecessor would exceed RP limit by providing
  // a reload as PHI incoming value
  LLVM_DEBUG(dbgs() << "  Checking predecessor RP (KillBB=" << printMBBReference(*KillBB) 
                    << ", RPLimit=" << RPLimit << "):\n");
  for (MachineBasicBlock *Pred : PIdfBB->predecessors()) {
    bool IsSpillPath = DT->dominates(KillBB, Pred);
    LLVM_DEBUG(dbgs() << "    " << printMBBReference(*Pred) 
                      << ": dominated by KillBB=" << (IsSpillPath ? "yes" : "no"));
    if (IsSpillPath) {
      // Spill-path predecessor - would need a reload at end
      // Check if there's room for the reload
      unsigned PredRP = getMaxRPForBlock(Pred);
      LLVM_DEBUG(dbgs() << ", RP=" << PredRP);
      if (PredRP > RPLimit) {
        LLVM_DEBUG(dbgs() << " > limit, REJECTING PHI\n");
        // Fall back to reload-at-use for all groups
        for (DomGroup *G : Groups) {
          insertReloadForUse(G->getHead(), SpilledVMP, KillBB);
        }
        return nullptr;
      }
      LLVM_DEBUG(dbgs() << " < limit, OK\n");
    } else {
      LLVM_DEBUG(dbgs() << "\n");
    }
  }

  // Insert PHI - empty IncomingValues means all predecessors use SpilledReg
  DenseMap<MachineBasicBlock *, Register> IncomingValues;
  
  MachineOperand *PHIResult = SSAUpdater->insertPHIAtBlock(
      PIdfBB, SpilledReg, IncomingValues, SpilledMask);
  Register PHIReg = PHIResult->getReg();
  MachineInstr *PHI = MRI->getVRegDef(PHIReg);
  
  LLVM_DEBUG(dbgs() << "  Inserted PHI: " << printReg(PHIReg, TRI) << "\n");
  
  SSAUpdater->rewriteDominatedUses(SpilledReg, PHIReg, SpilledMask);
  
  for (DomGroup *G : ReloadAtUseGroups) {
    insertReloadForUse(G->getHead(), SpilledVMP, KillBB);
  }
  
  return PHI;
}

void AMDGPUSSARegisterSpiller::processKillDominatedGroupsWithList(
    const SmallVectorImpl<DomGroup *> &Groups,
    VRegMaskPair SpilledVMP,
    MachineBasicBlock *KillBB,
    MachineInstr *KillMI,
    unsigned RPLimit) {
  
  Register SpilledReg = SpilledVMP.getVReg();
  LaneBitmask SpilledMask = SpilledVMP.getLaneMask();
  
  SmallVector<MachineInstr *, 4> Heads;
  for (DomGroup *G : Groups) {
    Heads.push_back(G->getHead());
  }
  
  if (!DisableReloadOptimizer && Heads.size() > 1) {
    auto ReloadPoints = optimizeReloadPlacing(Heads, RPLimit, SpilledReg);
    for (auto &RP : ReloadPoints) {
      auto Adjusted = adjustReloadForLoop(RP.first, RP.second, KillBB, SpilledReg);
      MachineBasicBlock *ReloadBB = Adjusted.first;
      
      MachineInstr *InsertBeforeHead = nullptr;
      for (MachineInstr *H : Heads) {
        if (H->getParent() == ReloadBB) {
          if (!InsertBeforeHead || 
              LIS->getInstructionIndex(*H) < LIS->getInstructionIndex(*InsertBeforeHead))
            InsertBeforeHead = H;
        }
      }
      
      auto [NewVReg, ReloadMI] = getOrCreateReloadInBlock(ReloadBB, SpilledVMP, InsertBeforeHead);
      if (ReloadMI) {
        // New reload created - use full SSA repair with IDF PHI insertion
        SSAUpdater->repairSSAForReload(NewVReg, SpilledReg, SpilledMask,
                                       ReloadMI->getParent());
      } else {
        // Cached reload - just rewrite dominated uses
        SSAUpdater->rewriteDominatedUses(SpilledReg, NewVReg, SpilledMask);
      }
    }
  } else {
    // Individual reloads path (no optimizer)
    for (MachineInstr *Head : Heads) {
      if (!usesSpilledVMP(Head, SpilledVMP))
        continue;
      insertReloadForUse(Head, SpilledVMP, KillBB);
    }
    
    // Fix pathological PHIs that still reference spilled register.
    // Only needed for individual reloads path - reload optimizer handles this via IDF PHIs.
    int FI = assignVirt2StackSlot(SpilledVMP);
    fixPathologicalPHIs(SpilledVMP, FI, KillMI);
  }
}

void AMDGPUSSARegisterSpiller::processKillDominatedGroups(
    SpillInfo &Info,
    MachineBasicBlock *KillBB,
    unsigned RPLimit) {
  
  MachineInstr *KillMI = Indexes->getInstructionFromIndex(Info.KillIdx);
  assert(KillMI && "KillIdx must correspond to an instruction");
  
  SmallVector<DomGroup *, 4> AllGroups;
  for (DomGroup &G : Info.DomGroups) {
    AllGroups.push_back(&G);
  }
  processKillDominatedGroupsWithList(AllGroups, Info.SpilledVMP, KillBB, KillMI, RPLimit);
}

void AMDGPUSSARegisterSpiller::finalizeLiveIntervals(Register SpilledReg) {
  // Reload intervals may already exist if they were created by repairSSAForReload's
  // internal call to performSSARepair (which uses LIS.getInterval() auto-creation).
  // Only create intervals for reloads that weren't processed through that path.
  for (auto &Entry : BlockReloadCache) {
    Register ReloadReg = Entry.second;
    if (!LIS->hasInterval(ReloadReg))
      LIS->createAndComputeVirtRegInterval(ReloadReg);
  }
  
  // SpilledReg always has an interval - recompute it to reflect rewritten uses
  LIS->removeInterval(SpilledReg);
  LIS->createAndComputeVirtRegInterval(SpilledReg);
}

void AMDGPUSSARegisterSpiller::emitReloadsAndRepairSSA(SpillInfo &Info) {
  VRegMaskPair SpilledVMP = Info.SpilledVMP;
  Register SpilledReg = SpilledVMP.getVReg();
  LaneBitmask SpilledMask = SpilledVMP.getLaneMask();
  
  MaxRPCache.clear();
  BlockReloadCache.clear();
  
  MachineInstr *KillMI = Indexes->getInstructionFromIndex(Info.KillIdx);
  assert(KillMI && "KillIdx must correspond to an instruction");
  MachineBasicBlock *KillBB = KillMI->getParent();
  
  LLVM_DEBUG({
    dbgs() << "\n=== emitReloadsAndRepairSSA() [PHI-first] ===\n";
    dbgs() << "Spilled: " << printReg(SpilledReg, TRI) 
           << " mask " << PrintLaneMask(SpilledMask) << "\n";
    dbgs() << "KillBB: " << printMBBReference(*KillBB) << "\n";
    dbgs() << "DomGroups: " << Info.DomGroups.size() << "\n";
  });
  
  unsigned RPLimit = IsVGPRPass ? VGPRLimit : SGPRLimit;
  
  SmallVector<MachineBasicBlock *, 8> PIdfBlocks;
  SSAUpdater->getPrunedIDF(SpilledReg, SpilledMask, KillBB, PIdfBlocks);
  
  if (PIdfBlocks.empty()) {
    LLVM_DEBUG(dbgs() << "  No PIDF blocks - all uses dominated by kill\n");
    processKillDominatedGroups(Info, KillBB, RPLimit);
    finalizeLiveIntervals(SpilledReg);
    return;
  }
  
  sortByDominanceOrder(PIdfBlocks);
  
  LLVM_DEBUG({
    dbgs() << "  PIDF blocks (sorted): ";
    for (auto *BB : PIdfBlocks) dbgs() << printMBBReference(*BB) << " ";
    dbgs() << "\n";
  });
  
  // Classify uses: kill-dominated vs PIDF-dominated
  // Note: Kill-dominated uses cannot have a PIDF dominating them (by construction)
  SmallVector<DomGroup *, 4> KillDominatedGroups;
  DenseMap<MachineBasicBlock *, SmallVector<DomGroup *, 2>> PIdfToGroups;
  
  for (DomGroup &G : Info.DomGroups) {
    MachineInstr *Head = G.getHead();
    
    if (DT->dominates(KillMI, Head)) {
      // Kill-dominated: no PIDF can dominate this (by construction)
      KillDominatedGroups.push_back(&G);
    } else {
      // Reachable but not dominated by kill - find closest PIDF
      MachineBasicBlock *ClosestPIDF = findClosestDominatingPIDF(Head, PIdfBlocks);
      if (ClosestPIDF) {
        PIdfToGroups[ClosestPIDF].push_back(&G);
      }
    }
  }
  
  LLVM_DEBUG(dbgs() << "  KillDominated groups: " << KillDominatedGroups.size() << "\n");
  
  // Phase 1: Process PIDF blocks - insert PHIs with incoming = SpilledReg
  SmallVector<MachineInstr *, 4> InsertedPHIs;
  for (MachineBasicBlock *PIdfBB : PIdfBlocks) {
    auto It = PIdfToGroups.find(PIdfBB);
    if (It == PIdfToGroups.end() || It->second.empty())
      continue;
    
    if (MachineInstr *PHI = processPIdfBlock(PIdfBB, SpilledVMP, KillBB, It->second, RPLimit))
      InsertedPHIs.push_back(PHI);
  }
  
  // Phase 2: Process kill-dominated groups (reloads rewrite PHI incoming operands)
  if (!KillDominatedGroups.empty()) {
    processKillDominatedGroupsWithList(KillDominatedGroups, SpilledVMP, KillBB, KillMI, RPLimit);
  }
  
  // Phase 3: Finalize live intervals (including PHI results)
  // Note: PHI intervals may already exist if:
  //   - The PHI was reused by repairSSAForReload (existing PHI check in createPHIInBlock)
  //   - performSSARepair already created the interval for a newly inserted PHI
  for (MachineInstr *PHI : InsertedPHIs) {
    Register PHIReg = PHI->getOperand(0).getReg();
    if (!LIS->hasInterval(PHIReg))
      LIS->createAndComputeVirtRegInterval(PHIReg);
    LLVM_DEBUG({
      dbgs() << "  PHI interval for " << printReg(PHIReg, TRI) << ": ";
      LIS->getInterval(PHIReg).print(dbgs());
      dbgs() << "\n";
    });
  }
  finalizeLiveIntervals(SpilledReg);
  
  LLVM_DEBUG(dbgs() << "\nemitReloadsAndRepairSSA() complete\n");
}

// ============================================================================
// Primary spill/reload methods (forward iterators) - used for SSA repair
// ============================================================================

MachineInstr *AMDGPUSSARegisterSpiller::spillAtDefinition(VRegMaskPair VMP) {
  if (MachineInstr *Existing = StoredAtDefinition.lookup(VMP)) {
    LLVM_DEBUG({
      StringRef Name = MRI->getVRegName(VMP.getVReg());
      dbgs() << "spillAtDefinition(): Already stored ";
      if (!Name.empty())
        dbgs() << "%" << Name;
      else
        dbgs() << printReg(VMP.getVReg(), TRI);
      dbgs() << " at definition\n";
    });
    return Existing;
  }

  Register VReg = VMP.getVReg();
  LaneBitmask Mask = VMP.getLaneMask();
  
  LLVM_DEBUG({
    StringRef Name = MRI->getVRegName(VReg);
    dbgs() << "spillAtDefinition(): Storing ";
    if (!Name.empty())
      dbgs() << "%" << Name;
    else
      dbgs() << printReg(VReg, TRI);
    dbgs() << " with mask " << PrintLaneMask(Mask) << " right after definition\n";
  });

  // Find the definition point
  MachineInstr *DefMI = MRI->getVRegDef(VReg);
  if (!DefMI) {
    LLVM_DEBUG(dbgs() << "spillAtDefinition(): No definition found (live-in?)\n");
    return nullptr;
  }

  MachineBasicBlock *DefMBB = DefMI->getParent();
  MachineBasicBlock::iterator InsertAfter = std::next(DefMI->getIterator());

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
      dbgs() << "spillAtDefinition(): Storing subregister "
             << TRI->getSubRegIndexName(SubRegIdx) << " of ";
      if (!Name.empty())
        dbgs() << "%" << Name;
      else
        dbgs() << printReg(VReg, TRI);
      dbgs() << "\n";
    }
  });

  // Emit the store instruction right after definition with isKill=false
  // This ensures all lanes are stored when EXEC is full
  TII->storeRegToStackSlot(*DefMBB, InsertAfter, VReg, /*isKill=*/false, FI, RC,
                           TRI, VReg, MachineInstr::NoFlags, SubRegIdx);

  // Get the inserted store instruction
  MachineInstr &StoreMI = *std::prev(InsertAfter);
  
  // Update LiveIntervals
  LIS->InsertMachineInstrInMaps(StoreMI);

  // Mark this register as stored at definition
  StoredAtDefinition[VMP] = &StoreMI;

  LLVM_DEBUG(dbgs() << "spillAtDefinition(): Stored: " << StoreMI);
  ++NumSpills;

  return &StoreMI;
}

void AMDGPUSSARegisterSpiller::spillBefore(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertBefore,
                                           VRegMaskPair VMP) {
  Register VReg = VMP.getVReg();
  LaneBitmask Mask = VMP.getLaneMask();
  
  // Check if this register was already stored at definition
  if (StoredAtDefinition.count(VMP)) {
    LLVM_DEBUG({
      StringRef Name = MRI->getVRegName(VReg);
      dbgs() << "spillBefore(): Register ";
      if (!Name.empty())
        dbgs() << "%" << Name;
      else
        dbgs() << printReg(VReg, TRI);
      dbgs() << " already stored at definition, marking dead at real spill point\n";
    });

    // Get the SlotIndex for the "real spill" point (right before InsertBefore)
    SlotIndex KillIdx;
    if (InsertBefore == MBB.begin()) {
      // InsertBefore is at block start, use block start index
      KillIdx = Indexes->getMBBStartIdx(&MBB);
    } else {
      // InsertBefore points to an instruction, mark dead at its base index
      // (right before it executes)
      KillIdx = Indexes->getInstructionIndex(*InsertBefore).getBaseIndex();
    }

    // Prune the LiveInterval to mark register dead at this point
    if (LIS->hasInterval(VReg)) {
      LiveInterval &LI = LIS->getInterval(VReg);
      
      // Prune the main live range (cast LiveInterval to LiveRange*)
      LIS->pruneValue(*static_cast<LiveRange *>(&LI), KillIdx, nullptr);
      
      // Prune all subranges as well
      for (LiveInterval::SubRange &SR : LI.subranges()) {
        LIS->pruneValue(SR, KillIdx, nullptr);
      }
      
      // TEMPORARY DEBUG: Dump %0's LiveInterval AFTER pruning and compare with %2
      LLVM_DEBUG({
        if (InsertBefore != MBB.begin()) {
          dbgs() << "spillBefore(): %0 LiveInterval AFTER pruning: " << LI << "\n";
          
          // Find %2 again and check if %0 dies before %2 becomes live
          MachineInstr *DefMI = &*InsertBefore;
          Register VReg2;
          for (const MachineOperand &MO : DefMI->operands()) {
            if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
              VReg2 = MO.getReg();
              break;
            }
          }
          
          if (VReg2.isValid() && LIS->hasInterval(VReg2)) {
            LiveInterval &LI2 = LIS->getInterval(VReg2);
            SlotIndex Def2Idx = Indexes->getInstructionIndex(*DefMI).getRegSlot();
            
            // Check if %2 is live at KillIdx (where %0 dies)
            LiveQueryResult LRQ2 = LI2.Query(KillIdx);
            bool VReg2LiveAtKill = (LRQ2.valueIn() != nullptr);
            
            dbgs() << "spillBefore(): %2 defined at: " << Def2Idx << "\n";
            dbgs() << "spillBefore(): %0 dies at: " << KillIdx << "\n";
            dbgs() << "spillBefore(): %2 live at KillIdx (" << KillIdx << "): " 
                   << (VReg2LiveAtKill ? "YES" : "NO") << "\n";
            
            if (VReg2LiveAtKill && Def2Idx >= KillIdx) {
              dbgs() << "spillBefore(): WARNING: %0 dies BEFORE %2 becomes live!\n";
            } else {
              dbgs() << "spillBefore(): OK: %0 dies after %2 becomes live or %2 not live yet\n";
            }
          }
        }
      });
    }

    LLVM_DEBUG(dbgs() << "spillBefore(): Pruned LiveInterval at " << KillIdx << "\n");
    return;
  }

  // Original behavior: emit spill instruction
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
  
  // TEMPORARY: RP check on reload disabled during SSA spilling.
  // This check is invalid because:
  // 1. RP hasn't converged yet (more registers will be spilled later)
  // 2. Spilled register's LiveInterval isn't shrunk until after emitReloadsAndRepairSSA
  // 3. Reloads on clean paths increase RP (needs split-before-use optimization)
  // We validate final RP at the end of processFunction instead.
  
  // OLD CODE (commented out temporarily):
  /*
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
  */

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
  
  LLVM_DEBUG(dbgs() << "repairSSAForReload(): Repairing SSA for reload: " 
                    << *ReloadMI);

                    SmallVector<MachineOperand *> PHIRegDefOps;
                    // Use class member SSA updater (shares IDF cache with
                    // reachability analysis)
                    Register NewVReg = SSAUpdater->repairSSAForNewDef(
                        *ReloadMI, OrigVReg, PHIRegDefOps);

                    for (auto *PHIRegDefOp : PHIRegDefOps) {
                      ReloadedRegs.insert(VRegMaskPair(*PHIRegDefOp, TRI, MRI));
                    }

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
      LaneBitmask UseMask = VRegMaskPair(MO, TRI, MRI).getLaneMask();
      // Check if this use overlaps with the spilled lanes
      if ((UseMask & SpilledMask).any()) {
        return true;
      }
    }
  }
  
  return false;
}

bool AMDGPUSSARegisterSpiller::walkPathsToUses(
    MachineBasicBlock *StartBB,
    Register SpilledReg,
    llvm::function_ref<bool(MachineBasicBlock *, MachineInstr *)> IsBad,
    bool stopOnBad) const {
  
  const LiveInterval &LI = LIS->getInterval(SpilledReg);
  
  SmallPtrSet<MachineBasicBlock *, 8> Visited;
  SmallVector<MachineBasicBlock *, 8> Worklist(StartBB->successors());
  bool FoundBad = false;
  
  while (!Worklist.empty()) {
    MachineBasicBlock *BB = Worklist.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;
    
    // Skip blocks where spilled register is not live
    SlotIndex BBStart = Indexes->getMBBStartIdx(BB);
    if (!LI.liveAt(BBStart))
      continue;
    
    // Find first use of SpilledReg in this block (if any)
    MachineInstr *FirstUseMI = nullptr;
    for (MachineInstr &MI : *BB) {
      if (MI.readsRegister(SpilledReg, TRI)) {
        FirstUseMI = &MI;
        break;
      }
    }
    
    // Check predicate
    if (IsBad(BB, FirstUseMI)) {
      if (stopOnBad)
        return false;
      FoundBad = true;
    }
    
    // Continue to successors
    for (MachineBasicBlock *Succ : BB->successors())
      if (!Visited.count(Succ))
        Worklist.push_back(Succ);
  }
  
  return !FoundBad;
}

bool AMDGPUSSARegisterSpiller::blockHasUse(MachineBasicBlock *BB,
                                           VRegMaskPair SpilledVMP,
                                           MachineInstr *StopInstr) const {
  // If we need to stop mid-block, fall back to instruction scan
  if (StopInstr && StopInstr->getParent() == BB) {
    auto EndIt = StopInstr->getIterator();
    for (auto I = BB->begin(); I != EndIt; ++I) {
      if (usesSpilledVMP(&*I, SpilledVMP)) {
        LLVM_DEBUG(dbgs() << "  Found use on path at: " << *I);
        return true;
      }
    }
    return false;
  }

  // Otherwise use NextUse's precomputed block summary
  bool HasUse = NU->usedInBlock(*BB).overlaps(SpilledVMP);
  
  LLVM_DEBUG({
    if (HasUse)
      dbgs() << "  Block " << printMBBReference(*BB) 
             << " has use of " << printReg(SpilledVMP.getVReg(), TRI) 
             << " (via NextUseAnalysis)\n";
  });
  
  return HasUse;
}

MachineBasicBlock *AMDGPUSSARegisterSpiller::tryHoistSpillToNCD(
    MachineInstr *KillMI, VRegMaskPair SpilledVMP,
    const SmallVectorImpl<MachineInstr *> &ReachableUses) {
  
  if (ReachableUses.empty())
    return nullptr;
    
  MachineBasicBlock *KillBB = KillMI->getParent();
  
  // Find nearest common dominator (NCD) of kill point and all reachable uses
  MachineBasicBlock *NCD = KillBB;
  for (MachineInstr *UseMI : ReachableUses) {
    MachineBasicBlock *UseBB = UseMI->getParent();
    NCD = DT->findNearestCommonDominator(NCD, UseBB);
    if (!NCD) {
      LLVM_DEBUG(dbgs() << "  No common dominator found, cannot hoist\n");
      return nullptr;
    }
  }
  
  LLVM_DEBUG(dbgs() << "  NCD for hoisting: " << printMBBReference(*NCD) << "\n");
  
  // Don't hoist into a loop - spill point inside loop would require reload every iteration
  if (MLI->getLoopFor(NCD)) {
    LLVM_DEBUG(dbgs() << "  NCD is inside a loop, cannot hoist\n");
    return nullptr;
  }
  
  auto IsUnexpectedUse = [&](MachineBasicBlock *, MachineInstr *UseMI) -> bool {
    if (!UseMI)
      return false;  // No use in this block - OK
    return !llvm::is_contained(ReachableUses, UseMI);
  };
  
  if (!walkPathsToUses(NCD, SpilledVMP.getVReg(), IsUnexpectedUse)) {
    LLVM_DEBUG(dbgs() << "  Uses exist on NCD→Kill path, cannot hoist\n");
    return nullptr;
  }
  
  LLVM_DEBUG(dbgs() << "  No unexpected uses on path, hoisting spill to NCD\n");
  
  // Note: With "store at definition", the actual store is at definition point.
  // Here we're hoisting the "kill point" (where register dies) to NCD.
  // The store remains at definition, but we mark the register dead earlier at NCD.
  // This requires updating the LiveInterval to mark the register dead at NCD instead of KillMI.
  
  // Get NCD end index (before terminator)
  SlotIndex NCDIdx = Indexes->getMBBEndIdx(NCD).getBaseIndex();
  
  // Prune LiveInterval at NCD instead of KillMI
  Register VReg = SpilledVMP.getVReg();
  if (LIS->hasInterval(VReg)) {
    LiveInterval &LI = LIS->getInterval(VReg);
    LIS->pruneValue(*static_cast<LiveRange *>(&LI), NCDIdx, nullptr);
    for (LiveInterval::SubRange &SR : LI.subranges()) {
      LIS->pruneValue(SR, NCDIdx, nullptr);
    }
  }
  
  // Move virtual spill marker to NCD if enabled
  if (EnableVirtualSpillMarkers) {
    // Remove old marker from KillBB (if present)
    for (MachineInstr &MI : llvm::make_early_inc_range(*KillBB)) {
      if (MI.getOpcode() == AMDGPU::SI_VIRTUAL_SPILL_MARKER &&
          MI.getOperand(0).getImm() == (int64_t)SpilledVMP.getVReg().virtRegIndex() &&
          MI.getOperand(1).getImm() == (int64_t)SpilledVMP.getLaneMask().getAsInteger()) {
        LIS->RemoveMachineInstrFromMaps(MI);
        MI.eraseFromParent();
        break;
      }
    }
    insertVirtualSpillMarker(*NCD, NCD->getFirstTerminator(), SpilledVMP);
  }
  
  LLVM_DEBUG(dbgs() << "  Hoisted kill point to: " << printMBBReference(*NCD) << "\n");
  
  return NCD;
}

void AMDGPUSSARegisterSpiller::collectDominatedBlocks(
    MachineBasicBlock &SpillMBB, SmallVectorImpl<MachineBasicBlock *> &DomBBs) const{
  DomTreeNodeBase<MachineBasicBlock> *Node = DT->getNode(&SpillMBB);
  if (Node) {
    for (auto *DN : depth_first(Node)) {
      DomBBs.push_back(DN->getBlock());
    }
  }
}

void AMDGPUSSARegisterSpiller::cutFromLiveRange(LiveRange &LR, SlotIndex CutStart, SlotIndex CutEnd) {
  if (!(CutStart < CutEnd))
    return;

  SmallVector<LiveRange::Segment, 16> Segs(LR.segments.begin(),
                                           LR.segments.end());

  for (const auto &S : Segs) {
    if (S.end <= CutStart || CutEnd <= S.start)
      continue; // no overlap

    // Remove the original.
    LR.removeSegment(S);

    // Keep left piece if any: [S.start, CutStart)
    if (S.start < CutStart)
      LR.addSegment(LiveRange::Segment(S.start, CutStart, S.valno));

    // Keep right piece if any: [CutEnd, S.end)
    if (CutEnd < S.end)
      LR.addSegment(LiveRange::Segment(CutEnd, S.end, S.valno));
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
  IsVGPRPass = false;
  bool ChangedSGPR = processFunction(MF, SGPRLimit);

  // Pass 2: VGPR Spilling
  LLVM_DEBUG(dbgs() << "\n=== Pass 2: Processing VGPRs ===\n");
  IsVGPRPass = true;
  bool ChangedVGPR = processFunction(MF, VGPRLimit);

  LLVM_DEBUG(dbgs() << "\nAMDGPUSSARegisterSpiller: Completed processing "
                    << MF.getName() << "\n");
  LLVM_DEBUG(dbgs() << "Total spills: " << NumSpills << ", Total reloads: "
                    << NumReloads << "\n");

  // Dump final LiveIntervals state for testing/verification
  LLVM_DEBUG({
    dbgs() << "\n********** FINAL LIVE INTERVALS **********\n";
    LIS->print(dbgs());
  });

  // Return true if either pass made modifications
  return ChangedSGPR || ChangedVGPR;
}



// Create function for pass manager
MachineFunctionPass *llvm::createAMDGPUSSARegisterSpillerPass() {
  return new AMDGPUSSARegisterSpiller();
}

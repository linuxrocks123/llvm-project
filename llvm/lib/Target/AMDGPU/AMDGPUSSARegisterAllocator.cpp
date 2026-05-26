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

        // Width != current pass: if wider, it was colored in a previous
        // pass and must be marked occupied at its def point (not live-in,
        // born in this BB). If narrower, it hasn't been colored yet —
        // ColorMap.find() returns end(), nothing to mark.
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
      }

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

// === Main entry point ===

bool AMDGPUSSARegisterAllocator::runOnMachineFunction(MachineFunction &MF) {
  TRI = static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  MRI = &MF.getRegInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  RegClassInfo.runOnMachineFunction(MF);

  LLVM_DEBUG(dbgs() << "AMDGPUSSARegisterAllocator: Processing "
                    << MF.getName() << "\n");

  classifyVRegs();
  OccupiedRegUnits.clear();
  OccupiedRegUnits.resize(TRI->getNumRegUnits());
  ColorMap.clear();

  color();

  // TODO: SSA-Destruct (phi to copies/swaps)
  // TODO: Operand rewrite (vreg to physreg)

  return !ColorMap.empty();
}

MachineFunctionPass *llvm::createAMDGPUSSARegisterAllocatorPass() {
  return new AMDGPUSSARegisterAllocator();
}

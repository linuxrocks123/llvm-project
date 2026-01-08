#!/usr/bin/env python3
"""
Renumber virtual registers in a MIR file to be compact starting from %0.
Vregs are numbered in the order they are DEFINED (first occurrence as def),
which matches MIRParser's behavior when no registers: section is present.

Note: Embedded LLVM IR modules (--- | ... ...) are preserved unchanged.
In LLVM IR, %N are block labels, not virtual registers.
"""

import re
import sys

def renumber_mir_vregs(filename):
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split file into parts:
    # 1. Header/CHECK section (before first ---)
    # 2. Optional LLVM IR module (--- | ... ... ---)
    # 3. MIR body (after final ---)
    
    # Find the LLVM IR module section if present (between "--- |" and "..." or next "---")
    ir_module_start = content.find('\n--- |\n')
    ir_module_end = -1
    ir_module = ""
    
    if ir_module_start != -1:
        # IR module can end with "..." or just the next "---" line
        # Try "..." first
        dots_end = content.find('\n...\n', ir_module_start)
        # Find next "---" after IR module start
        next_sep = content.find('\n---\n', ir_module_start + 6)  # +6 to skip "--- |"
        
        if dots_end != -1 and (next_sep == -1 or dots_end < next_sep):
            # "..." comes before next "---", use it as terminator
            ir_module_end = dots_end + 5  # Include the "...\n"
        elif next_sep != -1:
            # No "..." or it's after next "---", IR ends at next "---"
            ir_module_end = next_sep + 1  # Include the \n before ---
        
        if ir_module_end != -1:
            ir_module = content[ir_module_start:ir_module_end]
            print(f"Found embedded LLVM IR module ({len(ir_module)} chars) - will preserve unchanged")
    
    # Determine header and MIR sections based on IR module presence
    if ir_module_start != -1 and ir_module_end != -1:
        # Header is everything before IR module
        header_section = content[:ir_module_start + 1]  # Include trailing \n
        # MIR section is everything after the IR module's closing ...
        mir_section = content[ir_module_end:]
    else:
        # No IR module - find the MIR separator ---
        mir_sep = content.find('\n---\n')
        if mir_sep == -1:
            mir_sep = content.find('\n---')
        if mir_sep == -1:
            print("Error: Could not find '---' MIR separator in file")
            return
        header_section = content[:mir_sep + 1]
        mir_section = content[mir_sep + 1:]
    
    # Find 'body:' in mir_section for definition analysis
    body_idx = mir_section.find('\nbody:')
    if body_idx == -1:
        body_idx = mir_section.find('body:')
    if body_idx == -1:
        print("Error: Could not find 'body:' in MIR section")
        return
    
    body_section = mir_section[body_idx:]
    
    # Find vreg definitions in order they appear in the body
    # A definition is %N: (with colon indicating register class assignment)
    def_pattern = re.compile(r'%(\d+):[a-zA-Z_0-9]+\s*[,=]')
    
    seen_defs = set()
    ordered_defs = []
    
    for match in def_pattern.finditer(body_section):
        vreg_id = int(match.group(1))
        if vreg_id not in seen_defs:
            seen_defs.add(vreg_id)
            ordered_defs.append(vreg_id)
    
    print(f"Found {len(ordered_defs)} vreg definitions in body")
    
    # Find all vregs in MIR section only (not IR module)
    all_vreg_pattern = re.compile(r'%(\d+)')
    all_vregs = set()
    for match in all_vreg_pattern.finditer(mir_section):
        all_vregs.add(int(match.group(1)))
    
    # Also check header for CHECK patterns
    for match in all_vreg_pattern.finditer(header_section):
        all_vregs.add(int(match.group(1)))
    
    # Also check registers: section { id: N, ...}
    reg_id_pattern = re.compile(r'\{\s*id:\s*(\d+),')
    for match in reg_id_pattern.finditer(mir_section):
        all_vregs.add(int(match.group(1)))
    
    # Find use-only vregs (not in ordered_defs)
    use_only = sorted(all_vregs - seen_defs)
    
    print(f"Found {len(use_only)} use-only vregs (undef PHI operands, etc)")
    
    # Create mapping: definitions get IDs 0, 1, 2... in definition order
    # Then use-only vregs get the remaining IDs
    old_to_new = {}
    next_id = 0
    
    for old_id in ordered_defs:
        old_to_new[old_id] = next_id
        next_id += 1
    
    for old_id in use_only:
        old_to_new[old_id] = next_id
        next_id += 1
    
    print(f"Total vregs: {len(old_to_new)}")
    print(f"New range: 0 to {next_id - 1}")
    
    # Check if already correctly numbered
    already_correct = all(old == new for old, new in old_to_new.items())
    if already_correct:
        print("Vregs are already numbered correctly, no changes needed.")
        return
    
    # Show some mappings
    changes = [(old, new) for old, new in old_to_new.items() if old != new]
    if changes:
        print(f"\nMappings that change ({len(changes)} total):")
        for old, new in changes[:20]:
            print(f"  %{old} -> %{new}")
        if len(changes) > 20:
            print(f"  ... and {len(changes) - 20} more")
    
    # Apply replacements to header and MIR sections ONLY (not IR module)
    def apply_vreg_replacements(text):
        result = text
        # First pass: replace all vreg IDs with placeholders
        for old_id in sorted(all_vregs, reverse=True):
            new_id = old_to_new[old_id]
            placeholder = f"%__VREG_{new_id}__"
            # Replace %N followed by non-digit
            pattern = f'%{old_id}(?![0-9])'
            result = re.sub(pattern, placeholder, result)
        
        # Second pass: replace placeholders with final IDs
        for new_id in range(next_id):
            placeholder = f"%__VREG_{new_id}__"
            result = result.replace(placeholder, f"%{new_id}")
        
        # Also update registers: section { id: N, ...}
        def replace_reg_id(match):
            old_id = int(match.group(1))
            new_id = old_to_new.get(old_id, old_id)
            return f'{{ id: {new_id},'
        
        result = re.sub(r'\{ id: (\d+),', replace_reg_id, result)
        return result
    
    new_header = apply_vreg_replacements(header_section)
    new_mir = apply_vreg_replacements(mir_section)
    
    # Reconstruct file: header + IR module (unchanged) + MIR
    new_content = new_header + ir_module + new_mir
    
    # Write output
    with open(filename, 'w') as f:
        f.write(new_content)
    
    print(f"\nSuccessfully renumbered vregs in {filename}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mir_file>")
        sys.exit(1)
    
    renumber_mir_vregs(sys.argv[1])

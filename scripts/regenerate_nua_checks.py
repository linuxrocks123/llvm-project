#!/usr/bin/env python3
"""
Regenerate CHECK patterns for NextUseAnalysis unit tests.

This script runs the NextUseAnalysis pass on each MIR test file and captures
the debug output, then updates the CHECK patterns in the MIR file.
"""

import os
import subprocess
import sys
import re

def get_analysis_output(llc_path, mir_file):
    """Run NextUseAnalysis and capture debug output."""
    cmd = [
        llc_path,
        '-mtriple=amdgcn',
        '-mcpu=gfx1200',
        '-run-pass=amdgpu-next-use',
        '-debug-only=amdgpu-next-use',
        mir_file,
        '-o', '/dev/null'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stderr

def extract_check_content(output):
    """Extract the NextUseAnalysis results section from debug output."""
    # Find the start of results
    start_marker = "=== NextUseAnalysis Results for"
    end_marker = "=== End NextUseAnalysis Results ==="
    
    start_idx = output.find(start_marker)
    if start_idx == -1:
        return None
    
    end_idx = output.find(end_marker, start_idx)
    if end_idx == -1:
        end_idx = len(output)
    else:
        end_idx += len(end_marker)
    
    return output[start_idx:end_idx]

def format_as_checks(content):
    """Convert analysis output to CHECK patterns."""
    lines = content.split('\n')
    check_lines = []
    
    for line in lines:
        # Skip empty lines
        stripped = line.strip()
        if not stripped:
            continue
        
        # Determine the appropriate CHECK directive
        if stripped.startswith('==='):
            check_lines.append(f'# CHECK-LABEL: {stripped}')
        elif stripped.startswith('---'):
            check_lines.append(f'# CHECK: {stripped}')
        elif stripped.startswith('Instr:'):
            check_lines.append(f'# CHECK: {stripped}')
            check_lines.append('# CHECK-NEXT: Next-use distances:')
        elif stripped.startswith('Next-use distances:'):
            # Already added by Instr: handler
            continue
        elif stripped.startswith('Vreg:'):
            check_lines.append(f'# CHECK: {stripped}')
        elif stripped.startswith('Block End Distances:'):
            check_lines.append(f'# CHECK: {stripped}')
        elif stripped.startswith('(no register'):
            check_lines.append(f'# CHECK: {stripped}')
        
    return '\n'.join(check_lines)

def update_mir_file(mir_file, new_checks):
    """Replace old CHECK patterns in MIR file with new ones."""
    with open(mir_file, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Find all header lines to preserve (everything before first CHECK line)
    # This includes: RUN lines, NOTE comments, CFG diagrams, empty lines, etc.
    header_lines = []
    mir_body_start = -1
    in_header = True
    
    for i, line in enumerate(lines):
        # Found start of MIR body (first ---)
        if line.startswith('---'):
            mir_body_start = i
            break
        
        # Skip existing CHECK lines but keep everything else in header
        if line.startswith('# CHECK'):
            in_header = False
            continue
        
        # Keep all header content before first CHECK
        if in_header:
            header_lines.append(line)
    
    # Clean up trailing empty lines from header
    while header_lines and header_lines[-1].strip() == '':
        header_lines.pop()
    
    # Get the MIR body (everything from first --- onward)
    mir_body = '\n'.join(lines[mir_body_start:])
    
    # Construct new file content
    new_content = '\n'.join(header_lines) + '\n\n\n\n'
    new_content += new_checks + '\n\n'
    new_content += mir_body
    
    # Write back
    with open(mir_file, 'w') as f:
        f.write(new_content)

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    llvm_root = os.path.dirname(script_dir)
    
    # Try to find llc
    llc_paths = [
        os.path.join(llvm_root, 'build', 'Debug', 'bin', 'llc'),
        os.path.join(llvm_root, 'build', 'bin', 'llc'),
        os.path.join(llvm_root, 'build', 'Release', 'bin', 'llc'),
    ]
    
    llc_path = None
    for p in llc_paths:
        if os.path.exists(p):
            llc_path = p
            break
    
    if not llc_path:
        print("Error: Could not find llc binary")
        sys.exit(1)
    
    # Test directory
    test_dir = os.path.join(llvm_root, 'llvm', 'test', 'CodeGen', 'AMDGPU', 'NextUseAnalysis')
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)
    
    # Process each MIR file
    mir_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.mir')])
    
    # If specific file requested
    if len(sys.argv) > 1:
        mir_files = [f for f in mir_files if any(arg in f for arg in sys.argv[1:])]
    
    for mir_file in mir_files:
        mir_path = os.path.join(test_dir, mir_file)
        print(f"Processing {mir_file}...")
        
        # Get analysis output
        output = get_analysis_output(llc_path, mir_path)
        
        # Extract results section
        results = extract_check_content(output)
        if not results:
            print(f"  Warning: No analysis results found for {mir_file}")
            continue
        
        # Format as CHECK patterns
        checks = format_as_checks(results)
        
        # Update the file
        update_mir_file(mir_path, checks)
        print(f"  Updated {mir_file}")
    
    print("\nDone! Run the tests to verify.")

if __name__ == '__main__':
    main()


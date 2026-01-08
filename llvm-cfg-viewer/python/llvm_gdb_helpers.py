#!/usr/bin/env python3
"""
GDB Python helpers for LLVM debugging
Adds custom commands: viewCFG, viewCFGOnly
"""

import gdb
import re
import subprocess
import os
import tempfile
import time
import stat


RECENT_DOT_WINDOW_SECS = 120
VIEWER_STUB_DIR = "/tmp/llvm_cfg_disable_viewer"
VIEWER_STUBS = [
    "xdg-open",
    "open",
    "gv",
    "Graphviz",
    "xdot",
    "xdot.py",
    "dotty",
]


def _ensure_viewer_stub_dir():
    os.makedirs(VIEWER_STUB_DIR, exist_ok=True)
    script = "#!/bin/sh\n# LLVM CFG helper stub\nexit 0\n"
    for name in VIEWER_STUBS:
        path = os.path.join(VIEWER_STUB_DIR, name)
        if not os.path.exists(path):
            with open(path, "w") as fh:
                fh.write(script)
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def _block_graph_viewer():
    """Temporarily prepend PATH with stub executables to suppress GUI viewers."""
    _ensure_viewer_stub_dir()
    original_path = os.environ.get("PATH", "")
    if original_path.startswith(VIEWER_STUB_DIR):
        return original_path
    os.environ["PATH"] = f"{VIEWER_STUB_DIR}:{original_path}" if original_path else VIEWER_STUB_DIR
    return original_path


def _restore_viewer_path(original_path):
    os.environ["PATH"] = original_path


def _find_recent_dot_file():
    cutoff = time.time() - RECENT_DOT_WINDOW_SECS
    newest_path = None
    newest_mtime = -1.0

    for root, dirs, files in os.walk("/tmp"):
        for name in files:
            if not name.endswith(".dot"):
                continue
            path = os.path.join(root, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime < cutoff:
                continue
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_path = path

    return newest_path


def find_machine_function_in_stack():
    """
    Search up the call stack for a frame containing MachineFunction.
    Returns (frame, var_name) tuple or (None, None) if not found.
    """
    frame = gdb.selected_frame()
    frame_num = 0
    
    while frame is not None:
        try:
            # Get frame info
            frame_name = frame.name()
            
            # Check if this is runOnMachineFunction
            if frame_name and 'runOnMachineFunction' in frame_name:
                # Try to find MF parameter
                try:
                    frame.select()
                    mf = gdb.parse_and_eval('MF')
                    print(f"✅ Found MF in frame #{frame_num}: {frame_name}")
                    return (frame, 'MF', frame_num)
                except:
                    pass
            
            # Also check for common variable names
            for var_name in ['MF', 'mf', 'F', 'Fn']:
                try:
                    frame.select()
                    val = gdb.parse_and_eval(var_name)
                    type_str = str(val.type)
                    if 'MachineFunction' in type_str:
                        print(f"✅ Found {var_name} in frame #{frame_num}: {frame_name}")
                        return (frame, var_name, frame_num)
                except:
                    pass
            
        except Exception as e:
            pass
        
        # Move to older frame
        frame = frame.older()
        frame_num += 1
    
    return (None, None, None)

class ViewCFGCommand(gdb.Command):
    """View MachineFunction CFG in browser
    
    Usage: 
        viewCFG <MachineFunction*>  - Use specific variable
        viewCFG                     - Auto-search stack for MachineFunction
    
    Example: 
        viewCFG MF
        viewCFG
    
    Calls MachineFunction::viewCFG() and automatically opens in browser
    """
    
    def __init__(self):
        super(ViewCFGCommand, self).__init__("viewCFG", gdb.COMMAND_USER)
    
    def invoke(self, arg, from_tty):
        original_frame = gdb.selected_frame()
        target_frame = None
        
        # If no argument, search the stack
        if not arg.strip():
            print("🔍 Searching stack for MachineFunction...")
            frame, var_name, frame_num = find_machine_function_in_stack()
            if frame is None:
                print("❌ Could not find MachineFunction in call stack")
                print("   Try: viewCFG <variable_name>")
                return
            arg = var_name
            target_frame = frame
        else:
            target_frame = original_frame
        
        # Ensure we're in the right frame
        try:
            target_frame.select()
        except:
            pass
        
        # Evaluate the expression and determine if we need to take address
        try:
            mf_value = gdb.parse_and_eval(arg)
            mf_type = mf_value.type
            
            # Check if it's a reference or pointer
            type_str = str(mf_type)
            if '*' in type_str:
                # Already a pointer
                call_expr = f"{arg}->viewCFG()"
            else:
                # It's a reference or value, need to take address
                call_expr = f"(&{arg})->viewCFG()"
                print(f"🔍 Auto-converting {arg} to pointer")
            
        except gdb.error as e:
            print(f"❌ Error evaluating '{arg}': {e}")
            # Restore original frame
            original_frame.select()
            return
        
        print(f"🔍 Calling {call_expr}...")
        
        # Redirect stderr to capture DOT file path
        # Create a temporary pipe to capture output
        stderr_backup = None
        dot_file_path = None
        
        original_path = _block_graph_viewer()
        try:
            # Call viewCFG() method
            result = gdb.execute(f"call (void){call_expr}", to_string=True)
            
            # viewCFG() writes to stderr, which GDB captures
            dot_file_path = _find_recent_dot_file()
            if dot_file_path:
                print(f"✅ Found CFG: {dot_file_path}")
            
            if not dot_file_path:
                print("⚠️  CFG generated but couldn't locate .dot file")
                print("   Check /tmp for recent .dot files")
                return
            
            # Convert using our script
            script_path = "/work/atimofee/sandbox/scripts/view-cfg.sh"
            if os.path.exists(script_path):
                result = subprocess.run(
                    [script_path, dot_file_path],
                    capture_output=True, text=True
                )
                print(result.stderr)
                if result.returncode != 0:
                    print(f"❌ Conversion failed: {result.stderr}")
            else:
                print(f"⚠️  Script not found: {script_path}")
                print(f"   Manually run: {script_path} {dot_file_path}")
                
        except gdb.error as e:
            print(f"❌ Error calling viewCFG(): {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            _restore_viewer_path(original_path)
            try:
                original_frame.select()
            except:
                pass


class ViewCFGOnlyCommand(gdb.Command):
    """View MachineFunction CFG without liveness info
    
    Usage:
        viewCFGOnly <MachineFunction*>  - Use specific variable
        viewCFGOnly                     - Auto-search stack for MachineFunction
    
    Example: 
        viewCFGOnly MF
        viewCFGOnly
    
    Calls MachineFunction::viewCFGOnly() and automatically opens in browser
    """
    
    def __init__(self):
        super(ViewCFGOnlyCommand, self).__init__("viewCFGOnly", gdb.COMMAND_USER)
    
    def invoke(self, arg, from_tty):
        original_frame = gdb.selected_frame()
        target_frame = None
        
        # If no argument, search the stack
        if not arg.strip():
            print("🔍 Searching stack for MachineFunction...")
            frame, var_name, frame_num = find_machine_function_in_stack()
            if frame is None:
                print("❌ Could not find MachineFunction in call stack")
                print("   Try: viewCFGOnly <variable_name>")
                return
            arg = var_name
            target_frame = frame
        else:
            target_frame = original_frame
        
        # Ensure we're in the right frame
        try:
            target_frame.select()
        except:
            pass
        
        # Evaluate the expression and determine if we need to take address
        try:
            mf_value = gdb.parse_and_eval(arg)
            mf_type = mf_value.type
            
            # Check if it's a reference or pointer
            type_str = str(mf_type)
            if '*' in type_str:
                # Already a pointer
                call_expr = f"{arg}->viewCFGOnly()"
            else:
                # It's a reference or value, need to take address
                call_expr = f"(&{arg})->viewCFGOnly()"
                print(f"🔍 Auto-converting {arg} to pointer")
            
        except gdb.error as e:
            print(f"❌ Error evaluating '{arg}': {e}")
            # Restore original frame
            original_frame.select()
            return
        
        print(f"🔍 Calling {call_expr}...")
        
        dot_file_path = None
        original_path = _block_graph_viewer()
        try:
            result = gdb.execute(f"call (void){call_expr}", to_string=True)
            
            dot_file_path = _find_recent_dot_file()
            if dot_file_path:
                print(f"✅ Found CFG: {dot_file_path}")
            
            if not dot_file_path:
                print("⚠️  CFG generated but couldn't locate .dot file")
                return
            
            script_path = "/work/atimofee/sandbox/scripts/view-cfg.sh"
            if os.path.exists(script_path):
                result = subprocess.run(
                    [script_path, dot_file_path],
                    capture_output=True, text=True
                )
                print(result.stderr)
                
        except gdb.error as e:
            print(f"❌ Error calling viewCFGOnly(): {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            _restore_viewer_path(original_path)
            try:
                original_frame.select()
            except:
                pass


# Register commands
ViewCFGCommand()
ViewCFGOnlyCommand()

print("✅ LLVM GDB helpers loaded")
print("   Commands: viewCFG [<var>]         - View CFG (auto-searches stack if no arg)")
print("            viewCFGOnly [<var>]     - View CFG structure only")


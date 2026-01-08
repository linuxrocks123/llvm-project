# Auto-load LLVM debugging helpers
python
import sys
sys.path.insert(0, '/work/atimofee/sandbox/scripts')
try:
    exec(open('/work/atimofee/sandbox/scripts/llvm-gdb-helpers.py').read())
except Exception as e:
    print(f"⚠️  Failed to load LLVM GDB helpers: {e}")
end

# Set output styling
set print pretty on
set pagination off

# Optional: Start CFG server on GDB launch
shell /work/atimofee/sandbox/scripts/view-cfg.sh





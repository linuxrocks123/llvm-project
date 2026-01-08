# LLVM CFG Viewer

A VSCode/Cursor extension for visualizing LLVM MachineFunction Control Flow Graphs (CFGs) during debugging sessions.

## Features

- **Integrated CFG Panel**: View CFG graphs directly in VSCode without leaving the IDE
- **Search in CFG**: Find basic blocks, registers, and instructions with text search
- **Auto-conversion**: Automatically converts DOT files to SVG when detected
- **GDB Support**: Works with GDB debugger
- **Zoom Controls**: Zoom in/out and reset for large graphs
- **External Viewer**: Option to open in external browser or system viewer

## Requirements

- **Graphviz**: The `dot` command must be available in PATH
  ```bash
  # Ubuntu/Debian
  sudo apt install graphviz
  
  # macOS
  brew install graphviz
  ```

## Installation

### From VSIX (Recommended for development)

1. Build the extension:
   ```bash
   cd llvm-cfg-viewer
   npm install
   npm run compile
   npm run package
   ```

2. Install the generated `.vsix` file:
   - Open VSCode/Cursor
   - Press `Ctrl+Shift+P` → "Extensions: Install from VSIX"
   - Select the `.vsix` file

### Development Mode

1. Open the `llvm-cfg-viewer` folder in VSCode
2. Press `F5` to launch Extension Development Host

## Usage

### During Debugging

1. Start a GDB debug session
2. Set a breakpoint where you have access to a `MachineFunction`
3. Use one of these commands:
   - `Ctrl+Shift+P` → "LLVM: View CFG"
   - `Ctrl+Shift+P` → "LLVM: View CFG (Structure Only)"

### Manual Conversion

If auto-detection doesn't work:
1. In debug console, run: `-exec viewCFG`
2. Use `Ctrl+Shift+P` → "LLVM: Convert Latest DOT File"

### Search

- Type search text in the search box (e.g., `bb.1`, `%vreg`, `S_MOV`)
- Press Enter or click "Find"
- Use "Next"/"Prev" to navigate matches
- Press Escape or "Clear" to reset

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `llvmCfg.viewerMode` | `panel` | Where to display: `panel`, `browser`, or `system` |
| `llvmCfg.graphvizPath` | `dot` | Path to Graphviz dot binary |
| `llvmCfg.watchDirectory` | `/tmp` | Directory to watch for DOT files |
| `llvmCfg.autoConvert` | `true` | Auto-convert DOT files when detected |
| `llvmCfg.maxGraphsPerFunction` | `10` | Max graphs to keep per function |

## Debugger Setup

### GDB (cppdbg)

Add to your `launch.json`:

```json
{
    "type": "cppdbg",
    "MIMode": "gdb",
    "setupCommands": [
        {
            "description": "Load LLVM CFG helpers",
            "text": "-exec python exec(open('${workspaceFolder}/scripts/llvm-gdb-helpers.py').read())",
            "ignoreFailures": true
        }
    ]
}
```

## Troubleshooting

### "dot: command not found"
Install Graphviz: `sudo apt install graphviz`

### No CFG appears
1. Check Debug Console for errors
2. Verify you're stopped at a point with `MachineFunction` in scope
3. Try manual conversion: "LLVM: Convert Latest DOT File"

## License

MIT

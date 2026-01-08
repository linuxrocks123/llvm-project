# Installation Guide

## Prerequisites

1. **Node.js and npm** (for building)
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   
   # Or use nvm (recommended)
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   nvm install 18
   nvm use 18
   ```

2. **Graphviz** (for DOT to SVG conversion)
   ```bash
   sudo apt install graphviz
   ```

## Build and Install

```bash
cd llvm-cfg-viewer

# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Package as VSIX
npm run package
```

This creates `llvm-cfg-viewer-0.1.0.vsix`.

## Install in VSCode/Cursor

### Option 1: Command Palette
1. Press `Ctrl+Shift+P`
2. Type "Extensions: Install from VSIX"
3. Select the `.vsix` file

### Option 2: Command Line
```bash
code --install-extension llvm-cfg-viewer-0.1.0.vsix
# or for Cursor
cursor --install-extension llvm-cfg-viewer-0.1.0.vsix
```

## Development Mode

To run the extension in development mode:

1. Open `llvm-cfg-viewer` folder in VSCode/Cursor
2. Press `F5` to launch Extension Development Host
3. The extension will be active in the new window

## Verify Installation

1. Open Command Palette (`Ctrl+Shift+P`)
2. Type "LLVM" - you should see:
   - LLVM: View CFG
   - LLVM: View CFG (Structure Only)
   - LLVM: Convert Latest DOT File
   - LLVM: Open CFG in External Browser
   - LLVM: Show CFG Panel

## Configure Debugger

### For GDB

Add to your `.vscode/launch.json` in the debug configuration:

```json
"setupCommands": [
    {
        "description": "Load LLVM CFG helpers",
        "text": "-exec source ${workspaceFolder}/scripts/llvm-gdb-helpers.py",
        "ignoreFailures": true
    }
]
```

## Troubleshooting

### Extension doesn't appear
- Restart VSCode/Cursor after installation
- Check Extensions panel for "LLVM CFG Viewer"

### Commands not working
- Ensure you're in a GDB debug session
- Check Debug Console for error messages
- Verify Graphviz is installed: `dot -V`

### TypeScript errors during build
```bash
# Clean and rebuild
rm -rf out node_modules
npm install
npm run compile
```

import * as vscode from 'vscode';
import { CFGPanel } from './cfgPanel';
import { DotWatcher } from './dotWatcher';
import { DotConverter } from './converter';
import { DebuggerIntegration } from './debugger/integration';

let cfgPanel: CFGPanel | undefined;
let dotWatcher: DotWatcher | undefined;
let dotConverter: DotConverter;
let debuggerIntegration: DebuggerIntegration;

export function activate(context: vscode.ExtensionContext) {
    console.log('LLVM CFG Viewer is now active');

    // Initialize converter
    dotConverter = new DotConverter(context);

    // Initialize debugger integration
    debuggerIntegration = new DebuggerIntegration(context);

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('llvmCfg.viewCFG', async () => {
            await handleViewCFG(context, false);
        }),

        vscode.commands.registerCommand('llvmCfg.viewCFGOnly', async () => {
            await handleViewCFG(context, true);
        }),

        vscode.commands.registerCommand('llvmCfg.convertLatest', async () => {
            await handleConvertLatest(context);
        }),

        vscode.commands.registerCommand('llvmCfg.openExternal', async () => {
            await handleOpenExternal(context);
        }),

        vscode.commands.registerCommand('llvmCfg.showPanel', () => {
            showPanel(context);
        })
    );

    // Initialize DOT file watcher
    const config = vscode.workspace.getConfiguration('llvmCfg');
    const watchDir = config.get<string>('watchDirectory', '/tmp');
    const autoConvert = config.get<boolean>('autoConvert', true);

    if (autoConvert) {
        dotWatcher = new DotWatcher(watchDir, async (dotPath) => {
            const svgPath = await dotConverter.convert(dotPath);
            if (svgPath && cfgPanel) {
                cfgPanel.loadSvg(svgPath);
            }
        });
        dotWatcher.start();
        context.subscriptions.push({ dispose: () => dotWatcher?.stop() });
    }

    // Listen for debug session starts to inject helpers
    context.subscriptions.push(
        vscode.debug.onDidStartDebugSession((session) => {
            debuggerIntegration.onDebugSessionStart(session);
        })
    );

    // Show welcome message
    vscode.window.showInformationMessage(
        'LLVM CFG Viewer active. Use "LLVM: View CFG" during debugging.'
    );
}

async function handleViewCFG(context: vscode.ExtensionContext, structureOnly: boolean) {
    const session = vscode.debug.activeDebugSession;
    if (!session) {
        vscode.window.showWarningMessage('No active debug session. Start debugging first.');
        return;
    }

    // Show panel
    showPanel(context);

    // Send command to debugger
    const success = await debuggerIntegration.sendViewCFGCommand(session, structureOnly);
    
    if (!success) {
        vscode.window.showErrorMessage('Failed to execute viewCFG command');
    }
}

async function handleConvertLatest(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('llvmCfg');
    const watchDir = config.get<string>('watchDirectory', '/tmp');

    const dotPath = await dotConverter.findLatestDot(watchDir);
    if (!dotPath) {
        vscode.window.showWarningMessage('No recent DOT file found');
        return;
    }

    const svgPath = await dotConverter.convert(dotPath);
    if (svgPath) {
        showPanel(context);
        cfgPanel?.loadSvg(svgPath);
        vscode.window.showInformationMessage(`Converted: ${svgPath}`);
    }
}

async function handleOpenExternal(context: vscode.ExtensionContext) {
    if (cfgPanel?.currentSvgPath) {
        const uri = vscode.Uri.file(cfgPanel.currentSvgPath);
        await vscode.env.openExternal(uri);
    } else {
        vscode.window.showWarningMessage('No CFG loaded. Run viewCFG first.');
    }
}

function showPanel(context: vscode.ExtensionContext) {
    if (cfgPanel) {
        cfgPanel.reveal();
    } else {
        cfgPanel = new CFGPanel(context.extensionUri);
        cfgPanel.onDidDispose(() => {
            cfgPanel = undefined;
        });
    }
}

export function deactivate() {
    if (dotWatcher) {
        dotWatcher.stop();
    }
    if (cfgPanel) {
        cfgPanel.dispose();
    }
}




import * as vscode from 'vscode';
import * as path from 'path';

export class DebuggerIntegration {
    private readonly _context: vscode.ExtensionContext;
    private _pythonHelpersPath: string;

    constructor(context: vscode.ExtensionContext) {
        this._context = context;
        this._pythonHelpersPath = path.join(context.extensionPath, 'python');
    }

    /**
     * Called when a debug session starts - can be used to inject helpers
     */
    public onDebugSessionStart(session: vscode.DebugSession): void {
        const debuggerType = this._getDebuggerType(session);
        console.log(`Debug session started: ${session.name}, type: ${debuggerType}`);
        
        if (debuggerType === 'gdb') {
            console.log(`GDB helper available at: ${this._getGdbHelperPath()}`);
        }
    }

    /**
     * Send viewCFG command to the active debugger
     */
    public async sendViewCFGCommand(
        session: vscode.DebugSession,
        structureOnly: boolean = false
    ): Promise<boolean> {
        const debuggerType = this._getDebuggerType(session);
        const command = structureOnly ? 'viewCFGOnly' : 'viewCFG';

        try {
            if (debuggerType === 'gdb') {
                return await this._sendGdbCommand(session, command);
            } else {
                vscode.window.showWarningMessage(
                    `CFG visualization only supported with GDB. Current debugger: ${debuggerType}`
                );
                return false;
            }
        } catch (error) {
            console.error(`Failed to send ${command} command:`, error);
            return false;
        }
    }

    /**
     * Determine debugger type from session configuration
     */
    private _getDebuggerType(session: vscode.DebugSession): 'gdb' | 'other' {
        const config = session.configuration;

        // Check MIMode for cppdbg
        if (config.type === 'cppdbg' && config.MIMode !== 'lldb') {
            return 'gdb';
        }

        return 'other';
    }

    /**
     * Send command to GDB via debug console
     */
    private async _sendGdbCommand(session: vscode.DebugSession, command: string): Promise<boolean> {
        // For GDB, use -exec prefix to send MI command
        const gdbCommand = `-exec ${command}`;
        
        try {
            // First ensure helpers are loaded
            await this._ensureGdbHelpersLoaded(session);

            // Send the viewCFG command
            await session.customRequest('evaluate', {
                expression: gdbCommand,
                context: 'repl'
            });

            return true;
        } catch (error) {
            console.error('GDB command failed:', error);
            
            // Try alternative: direct command in debug console
            vscode.debug.activeDebugConsole?.appendLine(`> ${gdbCommand}`);
            
            return false;
        }
    }

    /**
     * Ensure GDB Python helpers are loaded
     */
    private async _ensureGdbHelpersLoaded(session: vscode.DebugSession): Promise<void> {
        const helperPath = this._getGdbHelperPath();
        
        try {
            await session.customRequest('evaluate', {
                expression: `-exec python exec(open("${helperPath}").read())`,
                context: 'repl'
            });
        } catch {
            // May already be loaded, ignore errors
        }
    }

    /**
     * Get path to GDB helper script
     */
    private _getGdbHelperPath(): string {
        // First check extension's bundled path
        const bundledPath = path.join(this._pythonHelpersPath, 'llvm_gdb_helpers.py');
        
        // Also check workspace scripts directory
        const workspaceFolders = vscode.workspace.workspaceFolders;
        if (workspaceFolders) {
            const workspacePath = path.join(
                workspaceFolders[0].uri.fsPath,
                'scripts',
                'llvm-gdb-helpers.py'
            );
            // Prefer workspace version if exists
            return workspacePath;
        }

        return bundledPath;
    }

    /**
     * Get helper loading command for launch.json
     */
    public getGdbSetupCommands(): object[] {
        const helperPath = this._getGdbHelperPath();
        return [
            {
                description: 'Load LLVM CFG helpers',
                text: `-exec python exec(open("${helperPath}").read())`,
                ignoreFailures: true
            }
        ];
    }
}

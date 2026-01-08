import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { spawn } from 'child_process';

export class DotConverter {
    private readonly _context: vscode.ExtensionContext;
    private readonly _outputDir: string;
    private readonly _maxGraphsPerFunction: number;

    constructor(context: vscode.ExtensionContext) {
        this._context = context;
        this._outputDir = '/tmp/cfg_graphs';
        
        const config = vscode.workspace.getConfiguration('llvmCfg');
        this._maxGraphsPerFunction = config.get<number>('maxGraphsPerFunction', 10);

        // Ensure output directory exists
        if (!fs.existsSync(this._outputDir)) {
            fs.mkdirSync(this._outputDir, { recursive: true });
        }
    }

    /**
     * Convert a DOT file to SVG
     */
    public async convert(dotPath: string): Promise<string | undefined> {
        const config = vscode.workspace.getConfiguration('llvmCfg');
        const dotBinary = config.get<string>('graphvizPath', 'dot');

        // Extract function name from DOT file
        const funcName = await this._extractFunctionName(dotPath);
        const funcDir = path.join(this._outputDir, funcName);

        // Ensure function directory exists
        if (!fs.existsSync(funcDir)) {
            fs.mkdirSync(funcDir, { recursive: true });
        }

        // Generate output filename with timestamp
        const timestamp = this._formatTimestamp(new Date());
        const outputPath = path.join(funcDir, `cfg_${timestamp}.svg`);

        try {
            await this._runDot(dotBinary, dotPath, outputPath);
            
            // Cleanup old graphs
            this._cleanupOldGraphs(funcDir);

            console.log(`Converted: ${dotPath} -> ${outputPath}`);
            return outputPath;
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to convert DOT file: ${error}`);
            return undefined;
        }
    }

    /**
     * Find the most recent DOT file in a directory
     */
    public async findLatestDot(directory: string, maxAgeSeconds: number = 120): Promise<string | undefined> {
        const cutoff = Date.now() - (maxAgeSeconds * 1000);
        let newest: string | undefined;
        let newestTime = 0;

        try {
            const entries = fs.readdirSync(directory);
            for (const entry of entries) {
                if (!entry.endsWith('.dot')) {
                    continue;
                }

                const fullPath = path.join(directory, entry);
                try {
                    const stat = fs.statSync(fullPath);
                    const mtime = stat.mtimeMs;
                    if (mtime > cutoff && mtime > newestTime) {
                        newest = fullPath;
                        newestTime = mtime;
                    }
                } catch {
                    // Ignore files we can't stat
                }
            }
        } catch {
            // Directory doesn't exist or can't be read
        }

        return newest;
    }

    /**
     * Extract function name from DOT file content
     */
    private async _extractFunctionName(dotPath: string): Promise<string> {
        try {
            const content = fs.readFileSync(dotPath, 'utf-8');
            const lines = content.split('\n');

            // Look for "CFG for 'funcname' function" pattern
            for (const line of lines.slice(0, 10)) {
                // Pattern: digraph "CFG for 'funcname' function"
                const match = line.match(/CFG for '([^']+)'/);
                if (match) {
                    return this._sanitizeFunctionName(match[1]);
                }

                // Fallback: digraph "funcname"
                const digestMatch = line.match(/digraph\s+"?([^"{\s]+)/);
                if (digestMatch && digestMatch[1] !== 'CFG') {
                    return this._sanitizeFunctionName(digestMatch[1]);
                }
            }
        } catch {
            // Ignore errors
        }

        return 'unknown_function';
    }

    /**
     * Sanitize function name for use as directory name
     */
    private _sanitizeFunctionName(name: string): string {
        // Remove special characters, keep alphanumeric and underscore
        let sanitized = name.replace(/[^a-zA-Z0-9_-]/g, '').toLowerCase();
        if (!sanitized) {
            sanitized = 'unknown_function';
        }
        return sanitized;
    }

    /**
     * Format timestamp for filename
     */
    private _formatTimestamp(date: Date): string {
        const pad = (n: number) => n.toString().padStart(2, '0');
        return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}_` +
               `${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
    }

    /**
     * Run Graphviz dot command
     */
    private _runDot(dotBinary: string, inputPath: string, outputPath: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const proc = spawn(dotBinary, ['-Tsvg', inputPath, '-o', outputPath]);

            let stderr = '';
            proc.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            proc.on('close', (code) => {
                if (code === 0) {
                    resolve();
                } else {
                    reject(new Error(`dot exited with code ${code}: ${stderr}`));
                }
            });

            proc.on('error', (err) => {
                reject(new Error(`Failed to run dot: ${err.message}`));
            });
        });
    }

    /**
     * Remove old graphs keeping only the most recent N
     */
    private _cleanupOldGraphs(funcDir: string): void {
        try {
            const files = fs.readdirSync(funcDir)
                .filter(f => f.endsWith('.svg'))
                .map(f => ({
                    name: f,
                    path: path.join(funcDir, f),
                    mtime: fs.statSync(path.join(funcDir, f)).mtimeMs
                }))
                .sort((a, b) => b.mtime - a.mtime);

            // Delete files beyond the limit
            const toDelete = files.slice(this._maxGraphsPerFunction);
            for (const file of toDelete) {
                fs.unlinkSync(file.path);
                console.log(`Cleaned up old graph: ${file.name}`);
            }
        } catch {
            // Ignore cleanup errors
        }
    }
}




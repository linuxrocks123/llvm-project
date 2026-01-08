import * as fs from 'fs';
import * as path from 'path';

export class DotWatcher {
    private readonly _watchDir: string;
    private readonly _callback: (dotPath: string) => void;
    private _watcher: fs.FSWatcher | undefined;
    private _processedFiles: Set<string> = new Set();
    private _debounceTimeout: NodeJS.Timeout | undefined;
    private _pendingFiles: Set<string> = new Set();

    constructor(watchDir: string, callback: (dotPath: string) => void) {
        this._watchDir = watchDir;
        this._callback = callback;
    }

    public start(): void {
        if (this._watcher) {
            return; // Already watching
        }

        try {
            this._watcher = fs.watch(this._watchDir, (eventType, filename) => {
                if (eventType === 'rename' && filename && filename.endsWith('.dot')) {
                    const fullPath = path.join(this._watchDir, filename);
                    this._handleDotFile(fullPath);
                }
            });

            console.log(`DOT watcher started on ${this._watchDir}`);
        } catch (error) {
            console.error(`Failed to start DOT watcher: ${error}`);
        }
    }

    public stop(): void {
        if (this._watcher) {
            this._watcher.close();
            this._watcher = undefined;
            console.log('DOT watcher stopped');
        }
        if (this._debounceTimeout) {
            clearTimeout(this._debounceTimeout);
        }
    }

    private _handleDotFile(dotPath: string): void {
        // Check if file exists (fs.watch fires on both create and delete)
        if (!fs.existsSync(dotPath)) {
            return;
        }

        // Get file modification time to create unique key
        let stat: fs.Stats;
        try {
            stat = fs.statSync(dotPath);
        } catch {
            return;
        }

        const fileKey = `${dotPath}:${stat.mtimeMs}`;
        if (this._processedFiles.has(fileKey)) {
            return; // Already processed this version
        }

        // Debounce to handle rapid writes
        this._pendingFiles.add(dotPath);

        if (this._debounceTimeout) {
            clearTimeout(this._debounceTimeout);
        }

        this._debounceTimeout = setTimeout(() => {
            this._processPendingFiles();
        }, 300); // Wait 300ms for file to be fully written
    }

    private _processPendingFiles(): void {
        const files = Array.from(this._pendingFiles);
        this._pendingFiles.clear();

        // Find the most recently modified file
        let newest: { path: string; mtime: number } | undefined;
        for (const filePath of files) {
            try {
                const stat = fs.statSync(filePath);
                if (!newest || stat.mtimeMs > newest.mtime) {
                    newest = { path: filePath, mtime: stat.mtimeMs };
                }
            } catch {
                continue;
            }
        }

        if (newest) {
            const fileKey = `${newest.path}:${newest.mtime}`;
            this._processedFiles.add(fileKey);

            // Limit cache size
            if (this._processedFiles.size > 100) {
                const entries = Array.from(this._processedFiles);
                this._processedFiles = new Set(entries.slice(-50));
            }

            console.log(`New DOT file detected: ${newest.path}`);
            this._callback(newest.path);
        }
    }
}




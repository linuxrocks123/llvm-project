import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

export class CFGPanel {
    public static readonly viewType = 'llvmCfgViewer';

    private readonly _panel: vscode.WebviewPanel;
    private readonly _extensionUri: vscode.Uri;
    private _disposables: vscode.Disposable[] = [];
    private _currentSvgPath: string | undefined;
    private _onDidDisposeEmitter = new vscode.EventEmitter<void>();

    public readonly onDidDispose = this._onDidDisposeEmitter.event;

    public get currentSvgPath(): string | undefined {
        return this._currentSvgPath;
    }

    constructor(extensionUri: vscode.Uri) {
        this._extensionUri = extensionUri;

        this._panel = vscode.window.createWebviewPanel(
            CFGPanel.viewType,
            'LLVM CFG Viewer',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [
                    vscode.Uri.file('/tmp'),
                    extensionUri
                ]
            }
        );

        this._panel.webview.html = this._getHtmlContent();

        this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

        this._panel.webview.onDidReceiveMessage(
            message => {
                switch (message.command) {
                    case 'openExternal':
                        if (this._currentSvgPath) {
                            vscode.env.openExternal(vscode.Uri.file(this._currentSvgPath));
                        }
                        break;
                    case 'info':
                        vscode.window.showInformationMessage(message.text);
                        break;
                }
            },
            null,
            this._disposables
        );
    }

    public reveal() {
        this._panel.reveal(vscode.ViewColumn.Beside);
    }

    public loadSvg(svgPath: string) {
        this._currentSvgPath = svgPath;

        try {
            const svgContent = fs.readFileSync(svgPath, 'utf-8');
            const funcName = this._extractFunctionName(svgPath);
            
            this._panel.webview.postMessage({
                command: 'loadSvg',
                svg: svgContent,
                functionName: funcName,
                path: svgPath
            });

            this._panel.title = `CFG: ${funcName}`;
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to load SVG: ${error}`);
        }
    }

    private _extractFunctionName(svgPath: string): string {
        // Extract from path like /tmp/cfg_graphs/functionname/cfg_timestamp.svg
        const parts = svgPath.split(path.sep);
        if (parts.length >= 2) {
            const funcDir = parts[parts.length - 2];
            if (funcDir !== 'cfg_graphs') {
                return funcDir;
            }
        }
        return path.basename(svgPath, '.svg');
    }

    public dispose() {
        this._onDidDisposeEmitter.fire();
        this._panel.dispose();

        while (this._disposables.length) {
            const x = this._disposables.pop();
            if (x) {
                x.dispose();
            }
        }
    }

    private _getHtmlContent(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data: https:;">
    <title>LLVM CFG Viewer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: var(--vscode-font-family);
            background: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        .header {
            background: var(--vscode-sideBar-background);
            padding: 8px 12px;
            border-bottom: 1px solid var(--vscode-panel-border);
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .title {
            font-weight: 600;
            font-size: 13px;
            color: var(--vscode-foreground);
            margin-right: auto;
        }
        .search-box {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .search-box input {
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            padding: 4px 8px;
            border-radius: 2px;
            font-size: 12px;
            width: 180px;
        }
        .search-box input:focus {
            outline: 1px solid var(--vscode-focusBorder);
        }
        button {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 4px 10px;
            border-radius: 2px;
            cursor: pointer;
            font-size: 11px;
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        button.secondary {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }
        .search-info {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
            margin-left: 8px;
        }
        .svg-container {
            flex: 1;
            overflow: auto;
            padding: 10px;
            background: var(--vscode-editor-background);
        }
        .svg-container svg {
            max-width: none;
            display: block;
        }
        .placeholder {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--vscode-descriptionForeground);
            font-size: 14px;
        }
        /* Search highlighting */
        .search-highlight-text {
            fill: #ffff00 !important;
            font-weight: bold !important;
        }
        .current-match {
            fill: #00ff00 !important;
            font-weight: bold !important;
            text-decoration: underline !important;
        }
        .zoom-controls {
            display: flex;
            gap: 4px;
            margin-left: 8px;
        }
        .status-bar {
            background: var(--vscode-statusBar-background);
            color: var(--vscode-statusBar-foreground);
            padding: 2px 8px;
            font-size: 11px;
            display: flex;
            justify-content: space-between;
        }
    </style>
</head>
<body>
    <div class="header">
        <span class="title" id="title">LLVM CFG Viewer</span>
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Search (e.g. bb.1, %vreg)" />
            <button onclick="doSearch()">Find</button>
            <button class="secondary" onclick="nextMatch()">Next</button>
            <button class="secondary" onclick="prevMatch()">Prev</button>
            <button class="secondary" onclick="clearSearch()">Clear</button>
        </div>
        <span class="search-info" id="searchInfo"></span>
        <div class="zoom-controls">
            <button onclick="zoomIn()">+</button>
            <button onclick="zoomOut()">-</button>
            <button onclick="zoomReset()">Reset</button>
        </div>
        <button onclick="openExternal()">Open External</button>
    </div>
    <div class="svg-container" id="svgContainer">
        <div class="placeholder">
            <span>Run "LLVM: View CFG" during a debug session to visualize CFG</span>
        </div>
    </div>
    <div class="status-bar">
        <span id="statusLeft">Ready</span>
        <span id="statusRight"></span>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        let allMatches = [];
        let currentMatchIndex = -1;
        let currentZoom = 1;
        let currentPath = '';

        // Handle messages from extension
        window.addEventListener('message', event => {
            const message = event.data;
            switch (message.command) {
                case 'loadSvg':
                    loadSvgContent(message.svg, message.functionName, message.path);
                    break;
            }
        });

        // Handle Enter key in search
        document.getElementById('searchInput').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                if (e.shiftKey) {
                    prevMatch();
                } else {
                    doSearch();
                }
            } else if (e.key === 'Escape') {
                clearSearch();
            }
        });

        function loadSvgContent(svgContent, functionName, filePath) {
            const container = document.getElementById('svgContainer');
            container.innerHTML = svgContent;
            currentPath = filePath;
            currentZoom = 1;
            
            document.getElementById('title').textContent = 'CFG: ' + functionName;
            document.getElementById('statusLeft').textContent = 'Loaded: ' + functionName;
            document.getElementById('statusRight').textContent = filePath;
            
            clearSearch();
            applyZoom();
        }

        function doSearch() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) return;

            clearHighlights();
            allMatches = [];
            currentMatchIndex = -1;

            const container = document.getElementById('svgContainer');
            const textElements = container.querySelectorAll('text');
            const queryLower = query.toLowerCase();

            textElements.forEach(el => {
                const text = el.textContent;
                if (text && text.toLowerCase().includes(queryLower)) {
                    allMatches.push(el);
                    el.classList.add('search-highlight-text');
                }
            });

            updateSearchInfo();
            if (allMatches.length > 0) {
                currentMatchIndex = 0;
                scrollToMatch(0);
            }
        }

        function nextMatch() {
            if (allMatches.length === 0) {
                doSearch();
                return;
            }
            currentMatchIndex = (currentMatchIndex + 1) % allMatches.length;
            scrollToMatch(currentMatchIndex);
        }

        function prevMatch() {
            if (allMatches.length === 0) return;
            currentMatchIndex = (currentMatchIndex - 1 + allMatches.length) % allMatches.length;
            scrollToMatch(currentMatchIndex);
        }

        function scrollToMatch(index) {
            allMatches.forEach(el => el.classList.remove('current-match'));
            
            const el = allMatches[index];
            el.classList.add('current-match');
            el.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
            updateSearchInfo();
        }

        function clearSearch() {
            clearHighlights();
            allMatches = [];
            currentMatchIndex = -1;
            document.getElementById('searchInfo').textContent = '';
        }

        function clearHighlights() {
            document.querySelectorAll('.search-highlight-text, .current-match').forEach(el => {
                el.classList.remove('search-highlight-text', 'current-match');
            });
        }

        function updateSearchInfo() {
            const info = document.getElementById('searchInfo');
            if (allMatches.length === 0) {
                info.textContent = 'No matches';
            } else {
                info.textContent = 'Match ' + (currentMatchIndex + 1) + ' of ' + allMatches.length;
            }
        }

        function zoomIn() {
            currentZoom = Math.min(currentZoom * 1.2, 5);
            applyZoom();
        }

        function zoomOut() {
            currentZoom = Math.max(currentZoom / 1.2, 0.2);
            applyZoom();
        }

        function zoomReset() {
            currentZoom = 1;
            applyZoom();
        }

        function applyZoom() {
            const svg = document.querySelector('#svgContainer svg');
            if (svg) {
                svg.style.transform = 'scale(' + currentZoom + ')';
                svg.style.transformOrigin = 'top left';
            }
        }

        function openExternal() {
            vscode.postMessage({ command: 'openExternal' });
        }
    </script>
</body>
</html>`;
    }
}




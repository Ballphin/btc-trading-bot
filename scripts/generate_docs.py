#!/usr/bin/env python3
"""Generate self-contained HTML documentation for TradingAgents.

This script analyzes the codebase and produces a single HTML file
with embedded CSS, JavaScript, and Mermaid diagrams that works
locally without external dependencies.

Usage:
    python scripts/generate_docs.py
    # Outputs: docs/tradingagents-docs.html
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "docs"
OUTPUT_FILE = OUTPUT_DIR / "tradingagents-docs.html"


def extract_docstring(node: ast.AST) -> str:
    """Extract docstring from AST node."""
    doc = ast.get_docstring(node)
    return doc or ""


def parse_module_structure() -> dict[str, Any]:
    """Walk tradingagents/ and extract module metadata."""
    modules = []
    tradingagents_dir = REPO_ROOT / "tradingagents"

    for py_file in sorted(tradingagents_dir.rglob("*.py")):
        if py_file.name.startswith("__"):
            continue

        rel_path = py_file.relative_to(REPO_ROOT)
        module_name = str(rel_path.with_suffix("")).replace("/", ".")

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)
            docstring = extract_docstring(tree)

            # Extract classes
            classes = []
            functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = extract_docstring(node)
                    methods = [
                        n.name for n in node.body
                        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
                    ]
                    classes.append({
                        "name": node.name,
                        "docstring": class_doc[:200] if class_doc else "",
                        "methods": methods[:5]  # Top 5 public methods
                    })
                elif isinstance(node, ast.FunctionDef) and node.name.startswith("_") is False:
                    func_doc = extract_docstring(node)
                    functions.append({
                        "name": node.name,
                        "docstring": func_doc[:150] if func_doc else "",
                        "signature": _get_signature(node)
                    })

            modules.append({
                "name": module_name,
                "path": str(rel_path),
                "docstring": docstring[:300] if docstring else "",
                "classes": classes[:3],  # Top 3 classes
                "functions": functions[:5]  # Top 5 functions
            })
        except SyntaxError:
            continue

    return {"modules": modules}


def _get_signature(node: ast.FunctionDef) -> str:
    """Build function signature from AST."""
    args = []
    for arg in node.args.args:
        arg_name = arg.arg
        if arg.annotation:
            if isinstance(arg.annotation, ast.Name):
                arg_name += f": {arg.annotation.id}"
            elif isinstance(arg.annotation, ast.Constant):
                arg_name += f": {arg.annotation.value}"
        args.append(arg_name)
    return f"({', '.join(args)})"


def extract_api_endpoints() -> dict[str, Any]:
    """Parse server.py for FastAPI endpoints using regex (more reliable for this use case)."""
    endpoints = []
    server_file = REPO_ROOT / "server.py"

    with open(server_file, "r", encoding="utf-8") as f:
        source = f.read()

    # Pattern to match FastAPI decorators and extract method, path, function name, and docstring
    # Matches: @app.get("/api/path") or @app.post("/api/path")
    pattern = r'@app\.(get|post|put|delete)\(["\']([^"\']+)["\']\)\s*\n\s*async?\s+def\s+(\w+)\s*\([^)]*\):\s*\n\s*"""([^"]*)"""'

    for match in re.finditer(pattern, source, re.IGNORECASE):
        method, path, func_name, docstring = match.groups()
        endpoints.append({
            "method": method.upper(),
            "path": path,
            "name": func_name,
            "docstring": docstring[:200] if docstring else ""
        })

    # Also try single-line docstrings
    pattern2 = r"@app\.(get|post|put|delete)\([\"\']([^\"\']+)[\"\']\)\s*\n\s*async?\s+def\s+(\w+)\s*\([^)]*\):\s*\n\s*'([^']*)'"

    for match in re.finditer(pattern2, source, re.IGNORECASE):
        method, path, func_name, docstring = match.groups()
        endpoints.append({
            "method": method.upper(),
            "path": path,
            "name": func_name,
            "docstring": docstring[:200] if docstring else ""
        })

    # Fallback: just extract all @app decorator lines with function names
    if not endpoints:
        lines = source.split('\n')
        current_decorator = None
        for i, line in enumerate(lines):
            dec_match = re.match(r'@app\.(get|post|put|delete)\(["\']([^"\']+)["\']\)', line, re.IGNORECASE)
            if dec_match:
                current_decorator = (dec_match.group(1).upper(), dec_match.group(2))
            elif current_decorator and 'async def ' in line:
                func_match = re.search(r'async def (\w+)', line)
                if func_match:
                    method, path = current_decorator
                    # Try to get docstring from next few lines
                    docstring = ""
                    for j in range(i+1, min(i+5, len(lines))):
                        doc_match = re.search(r'"""(.+?)"""', lines[j])
                        if doc_match:
                            docstring = doc_match.group(1)
                            break
                    endpoints.append({
                        "method": method,
                        "path": path,
                        "name": func_match.group(1),
                        "docstring": docstring[:200]
                    })
                    current_decorator = None

    return {"endpoints": endpoints}


def extract_frontend_components() -> dict[str, Any]:
    """Walk frontend/src and extract React components."""
    components = []
    frontend_dir = REPO_ROOT / "frontend" / "src"

    if not frontend_dir.exists():
        return {"components": []}

    for tsx_file in sorted(frontend_dir.rglob("*.tsx")):
        rel_path = tsx_file.relative_to(frontend_dir)

        try:
            with open(tsx_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple regex extraction for component names
            export_matches = re.findall(r'export\s+(?:default\s+)?(?:function|const)\s+(\w+)', content)
            route_matches = re.findall(r'path=["\']([^"\']+)["\']', content)

            components.append({
                "name": tsx_file.stem,
                "path": str(rel_path),
                "exports": export_matches[:3],
                "routes": route_matches
            })
        except Exception:
            continue

    return {"components": components}


def extract_config() -> dict[str, Any]:
    """Extract configuration from default_config.py and .env."""
    config = []

    config_file = REPO_ROOT / "tradingagents" / "default_config.py"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract CONFIG dict
        dict_match = re.search(r'CONFIG\s*=\s*\{([^}]+)\}', content, re.DOTALL)
        if dict_match:
            dict_content = dict_match.group(1)
            # Extract key-value pairs
            pairs = re.findall(r'"(\w+)":\s*([^,\n]+)', dict_content)
            for key, value in pairs:
                config.append({
                    "key": key,
                    "default": value.strip(),
                    "description": "Configuration parameter"
                })

    return {"config": config}


def generate_html(data: dict[str, Any]) -> str:
    """Generate the complete self-contained HTML document."""

    # Inline CSS
    css = """
:root {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --border: #475569;
  --code-bg: #0f172a;
  --success: #22c55e;
  --warning: #f59e0b;
  --error: #ef4444;
}

.light-mode {
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-tertiary: #e2e8f0;
  --text-primary: #0f172a;
  --text-secondary: #64748b;
  --border: #cbd5e1;
  --code-bg: #f1f5f9;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
  display: flex;
  min-height: 100vh;
}

/* Sidebar */
#sidebar {
  width: 260px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  position: fixed;
  height: 100vh;
  overflow-y: auto;
  padding: 1.5rem;
}

#sidebar h1 {
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--accent);
  margin-bottom: 0.5rem;
}

#sidebar .subtitle {
  font-size: 0.75rem;
  color: var(--text-secondary);
  margin-bottom: 1.5rem;
}

.nav-section {
  margin-bottom: 1rem;
}

.nav-section-title {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-secondary);
  margin-bottom: 0.5rem;
}

.nav-link {
  display: block;
  padding: 0.5rem 0.75rem;
  color: var(--text-secondary);
  text-decoration: none;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  transition: all 0.15s;
}

.nav-link:hover, .nav-link.active {
  background: var(--bg-tertiary);
  color: var(--text-primary);
}

.nav-link.active {
  border-left: 2px solid var(--accent);
}

/* Main content */
#main {
  margin-left: 260px;
  flex: 1;
  padding: 2rem;
  max-width: 900px;
}

section {
  display: none;
}

section.active {
  display: block;
}

h2 {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--accent);
}

h3 {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 1.5rem 0 0.75rem;
  color: var(--accent);
}

h4 {
  font-size: 1rem;
  font-weight: 600;
  margin: 1.25rem 0 0.5rem;
}

p { margin-bottom: 1rem; color: var(--text-secondary); }

/* Cards */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
  margin: 1.5rem 0;
}

.card {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 1.25rem;
  transition: transform 0.15s, box-shadow 0.15s;
}

.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
}

.card h4 {
  margin-top: 0;
  color: var(--accent);
}

/* Code blocks */
pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 1rem;
  overflow-x: auto;
  margin: 1rem 0;
}

code {
  font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
  font-size: 0.875rem;
}

pre code {
  color: var(--text-primary);
}

:not(pre) > code {
  background: var(--bg-tertiary);
  padding: 0.125rem 0.375rem;
  border-radius: 0.25rem;
  color: var(--accent);
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  font-size: 0.875rem;
}

th, td {
  padding: 0.75rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}

th {
  background: var(--bg-secondary);
  font-weight: 600;
  color: var(--accent);
}

/* Badges */
.badge {
  display: inline-block;
  padding: 0.125rem 0.5rem;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.badge-get { background: var(--success); color: white; }
.badge-post { background: var(--warning); color: white; }
.badge-class { background: var(--accent); color: white; }

/* Search */
#search {
  width: 100%;
  padding: 0.5rem 0.75rem;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  color: var(--text-primary);
  margin-bottom: 1rem;
  font-size: 0.875rem;
}

#search:focus {
  outline: none;
  border-color: var(--accent);
}

/* Theme toggle */
#theme-toggle {
  position: fixed;
  top: 1rem;
  right: 1rem;
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: 0.375rem;
  padding: 0.5rem;
  cursor: pointer;
  color: var(--text-primary);
  z-index: 100;
}

/* Mermaid */
.mermaid {
  background: var(--bg-secondary);
  border-radius: 0.5rem;
  padding: 1rem;
  margin: 1rem 0;
}

/* Tree view */
.tree {
  font-family: monospace;
  font-size: 0.875rem;
  line-height: 1.8;
  color: var(--text-secondary);
}

.tree-line {
  display: block;
}

.tree-dir { color: var(--accent); font-weight: 600; }
.tree-file { color: var(--text-secondary); }

/* Details/Summary */
details {
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  margin: 0.5rem 0;
}

summary {
  padding: 0.75rem 1rem;
  cursor: pointer;
  font-weight: 600;
}

details[open] summary {
  border-bottom: 1px solid var(--border);
}

details > div {
  padding: 1rem;
}

/* Formula */
.formula {
  background: var(--bg-secondary);
  border-left: 3px solid var(--accent);
  padding: 1rem 1.25rem;
  margin: 1rem 0;
  font-family: 'Times New Roman', serif;
  font-style: italic;
  font-size: 1.1rem;
}

/* Responsive */
@media (max-width: 768px) {
  #sidebar {
    width: 100%;
    position: relative;
    height: auto;
  }
  #main {
    margin-left: 0;
  }
  body {
    flex-direction: column;
  }
}

/* Hero */
.hero {
  text-align: center;
  padding: 3rem 0;
}

.hero h1 {
  font-size: 3rem;
  font-weight: 800;
  margin-bottom: 1rem;
  background: linear-gradient(135deg, var(--accent), #8b5cf6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero p {
  font-size: 1.25rem;
  max-width: 600px;
  margin: 0 auto 2rem;
}

.quick-links {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}

.btn {
  padding: 0.75rem 1.5rem;
  border-radius: 0.5rem;
  text-decoration: none;
  font-weight: 600;
  transition: all 0.15s;
}

.btn-primary {
  background: var(--accent);
  color: white;
}

.btn-primary:hover {
  background: var(--accent-hover);
}

.btn-secondary {
  background: var(--bg-tertiary);
  color: var(--text-primary);
  border: 1px solid var(--border);
}

.btn-secondary:hover {
  background: var(--border);
}
"""

    # Inline JavaScript (minimal, for routing and theme)
    js = """
// SPA Router
function showSection(id) {
  document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  
  const section = document.getElementById(id);
  const link = document.querySelector(`[href="#${id}"]`);
  
  if (section) section.classList.add('active');
  if (link) link.classList.add('active');
  
  window.scrollTo(0, 0);
}

// Hash-based routing
window.addEventListener('hashchange', () => {
  const hash = window.location.hash.slice(1) || 'welcome';
  showSection(hash);
});

// Initial route
const initialHash = window.location.hash.slice(1) || 'welcome';
showSection(initialHash);

// Theme toggle
const themeToggle = document.getElementById('theme-toggle');
themeToggle.addEventListener('click', () => {
  document.documentElement.classList.toggle('light-mode');
  localStorage.setItem('theme', document.documentElement.classList.contains('light-mode') ? 'light' : 'dark');
});

// Load saved theme
if (localStorage.getItem('theme') === 'light') {
  document.documentElement.classList.add('light-mode');
}

// Simple search
const search = document.getElementById('search');
const searchIndex = [];

// Build search index
document.querySelectorAll('section h2, section h3, section p').forEach(el => {
  const section = el.closest('section');
  if (section) {
    searchIndex.push({
      text: el.textContent.toLowerCase(),
      section: section.id,
      title: el.textContent
    });
  }
});

search.addEventListener('input', (e) => {
  const query = e.target.value.toLowerCase();
  if (!query) return;
  
  const match = searchIndex.find(item => item.text.includes(query));
  if (match) {
    window.location.hash = match.section;
  }
});

// Initialize Mermaid
if (typeof mermaid !== 'undefined') {
  mermaid.initialize({
    startOnLoad: true,
    theme: document.documentElement.classList.contains('light-mode') ? 'default' : 'dark',
    themeVariables: {
      primaryColor: '#3b82f6',
      primaryTextColor: '#f1f5f9',
      primaryBorderColor: '#475569',
      lineColor: '#64748b',
      secondaryColor: '#1e293b',
      tertiaryColor: '#334155'
    }
  });
}
"""

    # Mermaid.js minified (CDN fallback inline would be too large, so we'll use a data URI or simplified approach)
    # Instead, we'll use pre-rendered SVGs for key diagrams and let others be text-based

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TradingAgents Documentation</title>
  <style>{css}</style>
</head>
<body>
  <button id="theme-toggle">🌙</button>
  
  <nav id="sidebar">
    <h1>TradingAgents</h1>
    <div class="subtitle">Multi-Agent LLM Trading Framework</div>
    
    <input type="text" id="search" placeholder="Search...">
    
    <div class="nav-section">
      <div class="nav-section-title">Overview</div>
      <a href="#welcome" class="nav-link active">Welcome</a>
      <a href="#architecture" class="nav-link">Architecture</a>
    </div>
    
    <div class="nav-section">
      <div class="nav-section-title">Guides</div>
      <a href="#user-guide" class="nav-link">User Guide</a>
      <a href="#developer" class="nav-link">Developer Guide</a>
    </div>
    
    <div class="nav-section">
      <div class="nav-section-title">Reference</div>
      <a href="#api" class="nav-link">API Reference</a>
      <a href="#quant" class="nav-link">Quant Reference</a>
    </div>
  </nav>
  
  <main id="main">
    {render_welcome_section()}
    {render_architecture_section()}
    {render_user_guide_section()}
    {render_developer_section(data)}
    {render_api_section(data)}
    {render_quant_section()}
  </main>
  
  <script>{js}</script>
</body>
</html>"""

    return html


def render_welcome_section() -> str:
    """Render the Welcome section."""
    return """
    <section id="welcome" class="active">
      <div class="hero">
        <h1>TradingAgents</h1>
        <p>A production-grade, multi-agent LLM framework for quantitative trading research. Deploy specialized AI agents that debate to reach consensus on trading decisions.</p>
        
        <div class="quick-links">
          <a href="#architecture" class="btn btn-primary">Architecture</a>
          <a href="#user-guide" class="btn btn-secondary">Quick Start</a>
          <a href="#api" class="btn btn-secondary">API Reference</a>
        </div>
      </div>
      
      <div class="card-grid">
        <div class="card">
          <h4>🎯 For Traders</h4>
          <p>Run comprehensive AI analysis on any ticker. Get structured trading signals with confidence scores, stop-losses, and take-profit levels.</p>
          <code>python main.py --ticker BTC-USD</code>
        </div>
        
        <div class="card">
          <h4>🔬 For Quants</h4>
          <p>Walk-forward validation, Deflated Sharpe Ratio, ensemble methods, and shadow trading for rigorous strategy testing.</p>
          <a href="#quant" style="color: var(--accent);">See Quant Reference →</a>
        </div>
        
        <div class="card">
          <h4>💻 For Developers</h4>
          <p>Multi-agent LangGraph orchestration, FastAPI backend, React dashboard. Add your own agents and data sources.</p>
          <a href="#developer" style="color: var(--accent);">Developer Guide →</a>
        </div>
      </div>
      
      <h3>Key Features</h3>
      <table>
        <tr>
          <th>Feature</th>
          <th>Description</th>
        </tr>
        <tr>
          <td><strong>Multi-Provider LLM</strong></td>
          <td>OpenAI, Anthropic, Google, xAI, OpenRouter, Ollama — swap models without changing code</td>
        </tr>
        <tr>
          <td><strong>Asset-Agnostic</strong></td>
          <td>Equities via yfinance/Alpha Vantage; Crypto via Hyperliquid/Coinbase/Deribit</td>
        </tr>
        <tr>
          <td><strong>Production Backtesting</strong></td>
          <td>Replay, simulation, and hybrid modes with funding rates, slippage, walk-forward validation</td>
        </tr>
        <tr>
          <td><strong>Shadow Trading</strong></td>
          <td>Paper-trade mode for forward-testing with Brier score calibration</td>
        </tr>
        <tr>
          <td><strong>4H Scheduler</strong></td>
          <td>Automated crypto analysis every 4 hours, synced to UTC candle closes</td>
        </tr>
        <tr>
          <td><strong>React Dashboard</strong></td>
          <td>Real-time SSE streaming, interactive charts, backtest visualization</td>
        </tr>
      </table>
      
      <h3>Quick Start</h3>
      <pre><code># 1. Clone and setup
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 3. Start the backend
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# 4. Start the frontend (new terminal)
cd frontend && npm install && npm run dev

# 5. Open http://localhost:5173</code></pre>
    </section>
"""


def render_architecture_section() -> str:
    """Render the Architecture section with diagrams."""
    return """
    <section id="architecture">
      <h2>System Architecture</h2>
      
      <h3>High-Level Data Flow</h3>
      <div class="mermaid">
flowchart TB
    User[User Browser] -->|HTTP/WebSocket| Frontend[React Dashboard]
    Frontend -->|POST /api/analyze| Backend[FastAPI Server]
    Backend -->|SSE Stream| Frontend
    Backend -->|Job ID| User
    
    Backend -->|Invoke| Graph[TradingAgentsGraph]
    Graph -->|Parallel| Analysts[4 Analyst Agents]
    
    subgraph Analysts
        MA[Market Analyst]
        SA[Sentiment Analyst]
        NA[News Analyst]
        FA[Fundamentals Analyst]
    end
    
    Analysts -->|Research Reports| Debate[Bull/Bear Debate]
    Debate -->|Consensus| Risk[Risk Manager]
    Risk -->|Sized Signal| Signal[Signal Output]
    
    Analysts -.->|Data Requests| Data[Data Clients]
    
    subgraph Data
        YF[yfinance]
        HL[Hyperliquid]
        CB[Coinbase]
        DR[Deribit]
        BN[Binance]
    end
      </div>
      
      <h3>Agent State Machine (LangGraph)</h3>
      <div class="mermaid">
stateDiagram-v2
    [*] --> ParallelAnalysts: START
    
    ParallelAnalysts --> Synthesize: 4 Reports Complete
    
    state ParallelAnalysts {
        [*] --> Market
        [*] --> Sentiment
        [*] --> News
        [*] --> Fundamentals
        Market --> [*]
        Sentiment --> [*]
        News --> [*]
        Fundamentals --> [*]
    }
    
    Synthesize --> Debate: Research Synthesis
    Debate --> Risk: Bull/Bear Conclusion
    Risk --> Signal: Position Sizing
    Signal --> [*]: OUTPUT
      </div>
      
      <h3>Module Structure</h3>
      <div class="tree">
<span class="tree-line"><span class="tree-dir">tradingagents/</span></span>
<span class="tree-line">├── <span class="tree-dir">agents/</span>          <span class="tree-file"># Analyst implementations (Market, Sentiment, News, Fundamentals)</span></span>
<span class="tree-line">│   ├── <span class="tree-file">market_analyst.py</span></span>
<span class="tree-line">│   ├── <span class="tree-file">sentiment_analyst.py</span></span>
<span class="tree-line">│   ├── <span class="tree-file">news_analyst.py</span></span>
<span class="tree-line">│   └── <span class="tree-file">fundamentals_analyst.py</span></span>
<span class="tree-line">├── <span class="tree-dir">backtesting/</span>      <span class="tree-file"># Walk-forward, shadow trading, scorecards</span></span>
<span class="tree-line">│   ├── <span class="tree-file">engine.py</span>        <span class="tree-file"># Replay & simulation</span></span>
<span class="tree-line">│   ├── <span class="tree-file">walk_forward.py</span>  <span class="tree-file"># Walk-forward validation</span></span>
<span class="tree-line">│   ├── <span class="tree-file">scorecard.py</span>     <span class="tree-file"># Shadow trading scoring</span></span>
<span class="tree-line">│   ├── <span class="tree-file">metrics.py</span>       <span class="tree-file"># Sharpe, DSR, etc.</span></span>
<span class="tree-line">│   └── <span class="tree-file">regime.py</span>        <span class="tree-file"># Market regime detection</span></span>
<span class="tree-line">├── <span class="tree-dir">dataflows/</span>       <span class="tree-file"># Exchange clients & data routing</span></span>
<span class="tree-line">│   ├── <span class="tree-file">hyperliquid_client.py</span></span>
<span class="tree-line">│   ├── <span class="tree-file">coinbase_client.py</span></span>
<span class="tree-line">│   ├── <span class="tree-file">deribit_client.py</span></span>
<span class="tree-line">│   ├── <span class="tree-file">binance_client.py</span></span>
<span class="tree-line">│   ├── <span class="tree-file">y_finance.py</span></span>
<span class="tree-line">│   └── <span class="tree-file">interface.py</span>     <span class="tree-file"># Vendor routing</span></span>
<span class="tree-line">├── <span class="tree-dir">graph/</span>           <span class="tree-file"># LangGraph orchestration</span></span>
<span class="tree-line">│   ├── <span class="tree-file">trading_graph.py</span>  <span class="tree-file"># Main graph definition</span></span>
<span class="tree-line">│   ├── <span class="tree-file">ensemble_orchestrator.py</span></span>
<span class="tree-line">│   └── <span class="tree-file">signal_processing.py</span></span>
<span class="tree-line">├── <span class="tree-dir">llm_clients/</span>     <span class="tree-file"># Multi-provider LLM support</span></span>
<span class="tree-line">│   ├── <span class="tree-file">openai_client.py</span></span>
<span class="tree-line">│   ├── <span class="tree-file">anthropic_client.py</span></span>
<span class="tree-line">│   ├── <span class="tree-file">google_client.py</span></span>
<span class="tree-line">│   └── <span class="tree-file">xai_client.py</span></span>
<span class="tree-line">├── <span class="tree-dir">pulse/</span>           <span class="tree-file"># 4H scheduler & ensemble</span></span>
<span class="tree-line">│   ├── <span class="tree-file">scheduler.py</span></span>
<span class="tree-line">│   ├── <span class="tree-file">ensemble.py</span></span>
<span class="tree-line">│   └── <span class="tree-file">gist_sync.py</span></span>
<span class="tree-line">└── <span class="tree-dir">patterns/</span>        <span class="tree-file"># Technical analysis patterns</span></span>
<span class="tree-line">    ├── <span class="tree-file">reversal.py</span></span>
<span class="tree-line">    ├── <span class="tree-file">continuation.py</span></span>
<span class="tree-line">    └── <span class="tree-file">support_resistance.py</span></span>
      </div>
      
      <h3>Deployment Architecture</h3>
      <div class="mermaid">
flowchart LR
    subgraph Client
        Browser[Browser localhost:5173]
    end
    
    subgraph Server
        API[FastAPI localhost:8000]
        Graph[LangGraph]
        Sched[4H Scheduler]
    end
    
    subgraph DataSources
        YF[yfinance]
        HL[Hyperliquid]
        CB[Coinbase]
        AV[Alpha Vantage]
        KF[Kalshi]
    end
    
    subgraph LLMProviders
        OAI[OpenAI]
        ANT[Anthropic]
        GOO[Google]
        XAI[xAI]
    end
    
    Browser -->|/api/*| API
    API --> Graph
    Graph -->|Data| DataSources
    Graph -->|Inference| LLMProviders
    API --> Sched
      </div>
      
      <h3>Key Technologies</h3>
      <table>
        <tr><th>Layer</th><th>Technology</th><th>Purpose</th></tr>
        <tr><td>Frontend</td><td>React 18 + TypeScript + Vite</td><td>Dashboard UI with real-time streaming</td></tr>
        <tr><td>Backend</td><td>FastAPI + Uvicorn</td><td>REST API + SSE streaming</td></tr>
        <tr><td>Orchestration</td><td>LangGraph</td><td>Multi-agent workflow management</td></tr>
        <tr><td>LLM</td><td>OpenAI, Anthropic, Google, xAI</td><td>Research & reasoning agents</td></tr>
        <tr><td>Data</td><td>Hyperliquid, yfinance, etc.</td><td>Price, funding, sentiment data</td></tr>
        <tr><td>Testing</td><td>pytest</td><td>Unit and integration tests</td></tr>
      </table>
    </section>
"""


def render_user_guide_section() -> str:
    """Render the User Guide section."""
    return """
    <section id="user-guide">
      <h2>User Guide</h2>
      
      <h3>Running Your First Analysis</h3>
      
      <h4>Option 1: Command Line</h4>
      <pre><code>python main.py --ticker BTC-USD --date 2026-04-21</code></pre>
      <p>This runs a complete analysis and outputs the debate transcript and final signal to your terminal.</p>
      
      <h4>Option 2: Web Dashboard</h4>
      <ol>
        <li>Start both backend and frontend (see Quick Start)</li>
        <li>Navigate to <code>http://localhost:5173</code></li>
        <li>Enter a ticker (e.g., <code>BTC-USD</code> or <code>AAPL</code>)</li>
        <li>Click "Analyze" and watch real-time agent progress</li>
      </ol>
      
      <h3>Understanding the Output</h3>
      
      <div class="card-grid">
        <div class="card">
          <h4>📊 Debate Panel</h4>
          <p>See the Bull vs Bear arguments from each analyst agent. This surfaces conflicting viewpoints that a single model might miss.</p>
        </div>
        <div class="card">
          <h4>🎯 Signal Badge</h4>
          <p>Final consensus: BUY, SELL, COVER, or UNDERWEIGHT with confidence percentage.</p>
        </div>
        <div class="card">
          <h4>🛡️ Risk Check</h4>
          <p>Position size recommendation, stop-loss price, and take-profit levels.</p>
        </div>
      </div>
      
      <h3>Signal Types</h3>
      <table>
        <tr><th>Signal</th><th>Meaning</th><th>Context</th></tr>
        <tr><td><code>BUY</code></td><td>Open long position</td><td>Bullish consensus with risk acceptance</td></tr>
        <tr><td><code>SELL</code></td><td>Close long position</td><td>Exit signal for existing longs</td></tr>
        <tr><td><code>SHORT</code></td><td>Open short position</td><td>Bearish consensus (crypto only)</td></tr>
        <tr><td><code>COVER</code></td><td>Close short position</td><td>Exit signal for existing shorts</td></tr>
        <tr><td><code>OVERWEIGHT</code></td><td>Increase position size</td><td>Strong conviction with acceptable risk</td></tr>
        <tr><td><code>UNDERWEIGHT</code></td><td>Decrease position size</td><td>Risk reduction signal</td></tr>
      </table>
      
      <h3>Backtesting</h3>
      
      <details>
        <summary>Replay Mode (Use Historical Decisions)</summary>
        <div>
          <p>Re-run past analyses exactly as they were originally computed. Great for verifying historical performance.</p>
          <pre><code>GET /api/backtest/{job_id}?mode=replay</code></pre>
        </div>
      </details>
      
      <details>
        <summary>Simulation Mode (Test on Past Dates)</summary>
        <div>
          <p>Run fresh analysis on historical dates with data as-of that date. Tests strategy on unseen data.</p>
          <pre><code>POST /api/backtest
{
  "ticker": "BTC-USD",
  "dates": ["2024-01-01", "2024-02-01"],
  "mode": "simulation"
}</code></pre>
        </div>
      </details>
      
      <details>
        <summary>Hybrid Mode (Mixed)</summary>
        <div>
          <p>Use historical decisions when available, simulate missing dates. Balances speed and coverage.</p>
        </div>
      </details>
      
      <h3>Shadow Trading (Paper Trading)</h3>
      <p>Record decisions without real capital, then score them against actual outcomes:</p>
      <pre><code># Record a decision
POST /api/shadow/record
{
  "ticker": "BTC-USD",
  "signal": "BUY",
  "confidence": 0.75,
  "stop_loss": 60000,
  "take_profit": 75000
}

# Score all decisions for a ticker
GET /api/shadow/score/BTC-USD</code></pre>
    </section>
"""


def render_developer_section(data: dict[str, Any]) -> str:
    """Render the Developer Guide section."""
    modules_html = ""
    for mod in data.get("modules", [])[:20]:  # Limit to 20 modules
        classes = ", ".join([c["name"] for c in mod.get("classes", [])])
        functions = ", ".join([f["name"] for f in mod.get("functions", [])])
        
        modules_html += f"""
        <details>
          <summary><code>{mod['name']}</code> - {mod.get('docstring', '')[:60]}...</summary>
          <div>
            <p><strong>Path:</strong> {mod['path']}</p>
            {f'<p><strong>Classes:</strong> {classes}</p>' if classes else ''}
            {f'<p><strong>Functions:</strong> {functions}</p>' if functions else ''}
          </div>
        </details>
        """

    return f"""
    <section id="developer">
      <h2>Developer Guide</h2>
      
      <h3>Project Structure</h3>
      <p>The codebase is organized around the principle of <strong>separation of concerns</strong>:</p>
      
      <ul>
        <li><strong>Agents</strong> contain domain-specific analysis logic</li>
        <li><strong>Dataflows</strong> abstract exchange APIs behind unified interfaces</li>
        <li><strong>Graph</strong> orchestrates agent collaboration via LangGraph</li>
        <li><strong>Backtesting</strong> provides rigorous strategy validation</li>
        <li><strong>Pulse</strong> handles automation and ensemble management</li>
      </ul>
      
      <h3>Adding a New Analyst Agent</h3>
      
      <ol>
        <li><strong>Create the agent file</strong> in <code>tradingagents/agents/analysts/</code>:</li>
      </ol>
      
      <pre><code>from tradingagents.agents.base_analyst import BaseAnalyst

class MacroAnalyst(BaseAnalyst):
    '''Analyzes macroeconomic trends and their impact on asset prices.'''
    
    def __init__(self, config):
        super().__init__(config)
        self.domain = "macro"
    
    async def analyze(self, ticker: str, context: dict) -> dict:
        # Fetch macro data
        inflation = await self.get_inflation_data()
        rates = await self.get_fed_rates()
        
        # Generate research report
        report = await self.llm.generate(
            prompt=f"Analyze macro impact on {{ticker}} given inflation={{inflation}}, rates={{rates}}"
        )
        
        return {{
            "analyst": "MacroAnalyst",
            "sentiment": self.parse_sentiment(report),
            "confidence": self.calculate_confidence(report),
            "reasoning": report
        }}</code></pre>
      
      <ol start="2">
        <li><strong>Register in the graph</strong> in <code>tradingagents/graph/trading_graph.py</code>:</li>
      </ol>
      
      <pre><code>from tradingagents.agents.analysts.macro_analyst import MacroAnalyst

# Add to analyst initialization
self.analysts = [
    MarketAnalyst(config),
    SentimentAnalyst(config),
    NewsAnalyst(config),
    FundamentalsAnalyst(config),
    MacroAnalyst(config),  # NEW
]</code></pre>
      
      <ol start="3">
        <li><strong>Add tests</strong> in <code>tests/test_agents/</code>:</li>
      </ol>
      
      <pre><code>async def test_macro_analyze():
    analyst = MacroAnalyst(mock_config)
    result = await analyst.analyze("BTC-USD", {{}})
    assert "sentiment" in result
    assert "confidence" in result</code></pre>
      
      <h3>Adding a New Data Source</h3>
      
      <ol>
        <li><strong>Create the client</strong> inheriting from <code>BaseDataClient</code>:</li>
      </ol>
      
      <pre><code>from tradingagents.dataflows.base_client import BaseDataClient

class NewExchangeClient(BaseDataClient):
    '''Client for New Exchange API.'''
    
    def get_price(self, ticker: str) -> float:
        response = self._request(f"https://api.newexchange.com/price/{{ticker}}")
        return float(response["price"])
    
    def get_ohlcv(self, ticker: str, timeframe: str = "1d") -> pd.DataFrame:
        # Implementation here
        pass</code></pre>
      
      <ol start="2">
        <li><strong>Register in the router</strong> in <code>tradingagents/dataflows/interface.py</code>:</li>
      </ol>
      
      <pre><code>VENDOR_METHODS = {{
    "hyperliquid": HyperliquidClient(),
    "coinbase": CoinbaseClient(),
    "newexchange": NewExchangeClient(),  # NEW
}}</code></pre>
      
      <h3>Module Reference</h3>
      <p>Auto-generated from codebase analysis:</p>
      {modules_html}
      
      <h3>Testing Guide</h3>
      
      <pre><code># Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_backtesting/test_walk_forward.py -v

# Run with coverage
pytest tests/ --cov=tradingagents --cov-report=html

# Run integration tests
pytest test_phase3_integration.py -v</code></pre>
    </section>
"""


def render_api_section(data: dict[str, Any]) -> str:
    """Render the API Reference section."""
    endpoints = data.get("endpoints", [])
    
    endpoints_html = ""
    for ep in endpoints[:30]:  # Limit to 30 endpoints
        method_class = f"badge-{ep.get('method', 'GET').lower()}"
        endpoints_html += f"""
        <tr>
          <td><span class="badge {method_class}">{ep.get('method', 'GET')}</span></td>
          <td><code>{ep.get('path', '')}</code></td>
          <td>{ep.get('name', '')}</td>
          <td>{ep.get('docstring', '')[:80]}</td>
        </tr>
        """

    return f"""
    <section id="api">
      <h2>API Reference</h2>
      
      <h3>FastAPI Endpoints</h3>
      <p>Auto-generated from <code>server.py</code>:</p>
      
      <table>
        <tr>
          <th>Method</th>
          <th>Path</th>
          <th>Handler</th>
          <th>Description</th>
        </tr>
        {endpoints_html}
      </table>
      
      <h3>Key Classes</h3>
      
      <details>
        <summary><span class="badge badge-class">CLASS</span> <code>TradingAgentsGraph</code></summary>
        <div>
          <p>Main LangGraph orchestrator. Manages the parallel analyst phase, debate synthesis, and risk management.</p>
          <p><strong>Key Methods:</strong></p>
          <ul>
            <li><code>run(ticker, date)</code> - Execute full analysis workflow</li>
            <li><code>stream(ticker, date)</code> - Generator yielding progress updates</li>
            <li><code>get_state()</code> - Current graph state</li>
          </ul>
        </div>
      </details>
      
      <details>
        <summary><span class="badge badge-class">CLASS</span> <code>BaseAnalyst</code></summary>
        <div>
          <p>Abstract base class for all analyst agents. Defines the <code>analyze()</code> interface.</p>
          <p><strong>Subclasses:</strong> MarketAnalyst, SentimentAnalyst, NewsAnalyst, FundamentalsAnalyst</p>
        </div>
      </details>
      
      <details>
        <summary><span class="badge badge-class">CLASS</span> <code>BacktestEngine</code></summary>
        <div>
          <p>Handles replay, simulation, and hybrid backtesting modes.</p>
          <p><strong>Key Methods:</strong></p>
          <ul>
            <li><code>replay(job_id)</code> - Re-run historical analysis</li>
            <li><code>simulate(ticker, dates)</code> - Fresh analysis on past dates</li>
            <li><code>hybrid(ticker, dates)</code> - Mixed mode</li>
          </ul>
        </div>
      </details>
      
      <details>
        <summary><span class="badge badge-class">CLASS</span> <code>WalkForwardValidator</code></summary>
        <div>
          <p>Implements walk-forward cross-validation for strategy testing.</p>
          <p><strong>Key Methods:</strong></p>
          <ul>
            <li><code>validate(strategy, folds)</code> - Run walk-forward test</li>
            <li><code>compute_metrics(folds)</code> - Aggregate fold statistics</li>
          </ul>
        </div>
      </details>
      
      <details>
        <summary><span class="badge badge-class">CLASS</span> <code>ShadowScorecard</code></summary>
        <div>
          <p>Scores paper-trading decisions against actual outcomes.</p>
          <p><strong>Key Methods:</strong></p>
          <ul>
            <li><code>record_decision(ticker, signal)</code> - Log paper trade</li>
            <li><code>score(ticker)</code> - Compute Brier score, win rate</li>
            <li><code>calibrate()</code> - Adjust confidence thresholds</li>
          </ul>
        </div>
      </details>
      
      <h3>Data Client Hierarchy</h3>
      <div class="tree">
<span class="tree-line"><span class="tree-dir">BaseDataClient</span></span>
<span class="tree-line">├── <span class="tree-file">HyperliquidClient</span> <span class="tree-file"># Crypto perps, funding rates</span></span>
<span class="tree-line">├── <span class="tree-file">CoinbaseClient</span> <span class="tree-file"># Spot crypto</span></span>
<span class="tree-line">├── <span class="tree-file">DeribitClient</span> <span class="tree-file"># Options, futures</span></span>
<span class="tree-line">├── <span class="tree-file">BinanceClient</span> <span class="tree-file"># Sub-daily OHLCV</span></span>
<span class="tree-line">├── <span class="tree-file">YFinanceClient</span> <span class="tree-file"># Equities</span></span>
<span class="tree-line">├── <span class="tree-file">AlphaVantageClient</span> <span class="tree-file"># Fundamentals</span></span>
<span class="tree-line">├── <span class="tree-file">KalshiClient</span> <span class="tree-file"># Event contracts</span></span>
<span class="tree-line">└── <span class="tree-file">PolymarketClient</span> <span class="tree-file"># Prediction markets</span></span>
      </div>
    </section>
"""


def render_quant_section() -> str:
    """Render the Quant Reference section."""
    return """
    <section id="quant">
      <h2>Quantitative Reference</h2>
      
      <h3>Deflated Sharpe Ratio (DSR)</h3>
      <p>The Deflated Sharpe Ratio accounts for the fact that testing multiple strategies inflates the observed Sharpe ratio (multiple comparisons problem).</p>
      
      <div class="formula">
        DSR = Φ((SR − E[max SR]) / SE(SR))
      </div>
      
      <p>Where:</p>
      <ul>
        <li><strong>SR</strong>: Observed Sharpe ratio</li>
        <li><strong>E[max SR]</strong>: Expected maximum Sharpe under null (from n_strategy trials)</li>
        <li><strong>SE(SR)</strong>: Standard error of the Sharpe ratio</li>
        <li><strong>Φ</strong>: Standard normal CDF</li>
      </ul>
      
      <p>Implementation: <code>tradingagents/backtesting/walk_forward.py::compute_deflated_sharpe()</code></p>
      
      <h3>Sharpe Standard Error (Lo, 2002)</h3>
      <p>The standard formula assumes Gaussian returns. Lo (2002) provides a correction for skewness and kurtosis:</p>
      
      <div class="formula">
        SE = √[(1 + ½·SR² − γ₁·SR + (γ₂/4)·SR²) / n]
      </div>
      
      <p>Where:</p>
      <ul>
        <li><strong>γ₁</strong>: Skewness of returns</li>
        <li><strong>γ₂</strong>: Excess kurtosis (kurtosis − 3)</li>
        <li><strong>n</strong>: Number of observations</li>
      </ul>
      
      <p><strong>Why this matters for crypto:</strong> Bitcoin returns typically have skewness ≈ −1.5 and kurtosis ≈ 8. For SR=1.2, n=100:</p>
      <ul>
        <li>Gaussian SE: 0.104</li>
        <li>Fat-tail SE: 0.177 (+70% inflation)</li>
      </ul>
      
      <p>Implementation: <code>tradingagents/backtesting/stats.py::sharpe_standard_error()</code></p>
      
      <h3>Regime Detection</h3>
      <p>Market regimes are classified using price action relative to moving averages and volatility thresholds:</p>
      
      <table>
        <tr><th>Regime</th><th>Condition</th><th>Interpretation</th></tr>
        <tr><td>trending_up</td><td>price > SMA20 > SMA50</td><td>Bullish trend intact</td></tr>
        <tr><td>trending_down</td><td>price < SMA20 < SMA50</td><td>Bearish trend intact</td></tr>
        <tr><td>ranging</td><td>SMA20 ≈ SMA50</td><td>No clear direction</td></tr>
        <tr><td>volatile</td><td>σ(20d) > threshold</td><td>High uncertainty</td></tr>
      </table>
      
      <p><strong>Sub-daily support:</strong> For crypto, 1H and 4H intervals use Binance data. Equities fall back to daily.</p>
      
      <p>Implementation: <code>tradingagents/backtesting/regime.py::detect_regime_context()</code></p>
      
      <h3>Walk-Forward Validation</h3>
      <p>Walk-forward testing prevents lookahead bias by:</p>
      <ol>
        <li>Dividing history into n folds</li>
        <li>Training on fold i, testing on fold i+1</li>
        <li>Aggregating out-of-sample results</li>
      </ol>
      
      <div class="mermaid">
flowchart LR
    subgraph Fold1[Fold 1]
        T1[Train] --> Te1[Test]
    end
    subgraph Fold2[Fold 2]
        T2[Train] --> Te2[Test]
    end
    subgraph Fold3[Fold 3]
        T3[Train] --> Te3[Test]
    end
    
    Fold1 --> Fold2 --> Fold3
    
    Te1 --> Aggregate[Aggregate Metrics]
    Te2 --> Aggregate
    Te3 --> Aggregate
      </div>
      
      <p>Key metrics per fold:</p>
      <ul>
        <li>Sharpe ratio (gross and net of costs)</li>
        <li>Sharpe SE (with skew/kurtosis)</li>
        <li>Deflated Sharpe Ratio (DSR)</li>
        <li>Win rate, profit factor</li>
        <li>Maximum drawdown</li>
      </ul>
      
      <p>Implementation: <code>tradingagents/backtesting/walk_forward.py::WalkForwardValidator</code></p>
      
      <h3>Shadow Trading Scoring</h3>
      <p>Paper trades are scored using the <strong>Brier score</strong> for calibration:</p>
      
      <div class="formula">
        Brier = (1/N) Σ(pᵢ − oᵢ)²
      </div>
      
      <p>Where pᵢ is predicted confidence and oᵢ is outcome (1 for correct, 0 for wrong). Lower is better.</p>
      
      <p>Scoring tiers:</p>
      <ul>
        <li><strong>Excellent:</strong> Brier < 0.15</li>
        <li><strong>Good:</strong> Brier 0.15−0.25</li>
        <li><strong>Poor:</strong> Brier > 0.25 (overconfident)</li>
      </ul>
      
      <p>Implementation: <code>tradingagents/backtesting/scorecard.py::run_calibration_study()</code></p>
    </section>
"""


def main():
    """Main entry point."""
    print("🔍 Analyzing codebase structure...")
    data = {}
    data.update(parse_module_structure())
    data.update(extract_api_endpoints())
    data.update(extract_frontend_components())
    data.update(extract_config())

    print("🎨 Generating HTML documentation...")
    html = generate_html(data)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    file_size = OUTPUT_FILE.stat().st_size / 1024  # KB
    print(f"✅ Generated: {OUTPUT_FILE}")
    print(f"📊 Size: {file_size:.1f} KB")
    print(f"📦 Modules documented: {len(data.get('modules', []))}")
    print(f"🔗 API endpoints: {len(data.get('endpoints', []))}")
    print(f"\n🚀 Open in browser: file://{OUTPUT_FILE.absolute()}")


if __name__ == "__main__":
    main()

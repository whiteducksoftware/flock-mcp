# flock-mcp
Multiple MCP servers and tools for the agent framework Flock.

## Installation

This package has no main dependencies by design, allowing you to install only the tools you need. You can install it in several ways:

### Option 1: Install All Tools (Recommended)
```bash
pip install flock-mcp[all]
```
Or with uv:
```bash
uv add flock-mcp[all]
```
This installs all available tools and dependencies.

### Option 2: Install Specific Tool Groups
Choose the specific tool groups you need:

```bash
# Basic tools (web scraping, markdown, search)
pip install flock-mcp[basic-tools]
uv add flock-mcp[basic-tools]

# Azure integration tools
pip install flock-mcp[azure-tools]
uv add flock-mcp[azure-tools]

# LLM and text processing tools
pip install flock-mcp[llm-tools]
uv add flock-mcp[llm-tools]

# Code and Docker tools
pip install flock-mcp[code-tools]
uv add flock-mcp[code-tools]

# All tools (same as [all] but without extra ML dependencies)
pip install flock-mcp[all-tools]
uv add flock-mcp[all-tools]
```

### Option 3: Combine Multiple Groups
You can combine multiple optional dependency groups:

```bash
pip install flock-mcp[basic-tools,azure-tools]
uv add flock-mcp[basic-tools,azure-tools]
```

### Option 4: Base Installation Only
Install just the base package without any optional dependencies:

```bash
pip install flock-mcp
uv add flock-mcp
```

### Available Tool Groups

- **`basic-tools`**: Web scraping (docling, tavily-python), markdown processing (markdownify), search (duckduckgo-search)
- **`azure-tools`**: Azure cloud integration (azure-identity, azure-storage-blob, azure-search-documents)
- **`llm-tools`**: Natural language processing (nltk)
- **`code-tools`**: Code analysis and containerization (docker)
- **`all-tools`**: All of the above tool groups
- **`all`**: All tools plus additional ML/AI dependencies (datasets, rouge-score, sentence-transformers, zep-python, mem0ai, chromadb, matplotlib)

## Usage

After installation, you can use the various MCP servers and tools provided by this package with the Flock agent framework.

```python
# eg
from flock.tools.file_tools import file....
```

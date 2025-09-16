# Policy as Code CLI

A command-line tool to extract, consolidate, and validate policy rules from documents using AI.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Configure API
```bash
python policyascode.py config --api-key sk-... --base-url https://openrouter.ai/api/v1
```

### Extract rules from documents
```bash
python policyascode.py extract -o rules.json policy.md contract.pdf
```

### Consolidate duplicates
```bash
python policyascode.py consolidate -i rules.json -o consolidated.json
```

### Validate compliance
```bash
python policyascode.py validate -r consolidated.json document.md
```

## Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `extract` | Extract atomic rules from documents | `python policyascode.py extract -o rules.json policy.md` |
| `consolidate` | Remove duplicates and merge similar rules | `python policyascode.py consolidate -i rules.json -o clean.json` |
| `validate` | Check document compliance against rules | `python policyascode.py validate -r rules.json doc.md` |
| `config` | Set API credentials and preferences | `python policyascode.py config --api-key sk-...` |

## Global Options

- `--model <model>`: LLM model to use (default: openai/gpt-4o-mini)
- `--base-url <url>`: API base URL (default: https://openrouter.ai/api/v1)  
- `--api-key <key>`: API key (can also use OPENAI_API_KEY env var)

## Custom Prompts

- `--extraction-prompt <text>`: Custom extraction prompt
- `--consolidation-prompt <text>`: Custom consolidation prompt  
- `--validation-prompt <text>`: Custom validation prompt

## Supported Formats

- **Input**: Markdown (.md), Text (.txt), PDF (.pdf)
- **Output**: JSON with structured rule data

## Rule Structure

```json
{
  "id": "rule-1",
  "title": "Brief rule summary",
  "body": "Detailed rule description", 
  "priority": "high|medium|low",
  "rationale": "Why this rule exists",
  "source_file": "policy.md",
  "sources": [{"quote": "Original text"}]
}
```

## Configuration

Configuration is stored in `~/.policyascode/config` and can be set via:
- Command line: `policyascode config --api-key sk-...`
- Environment variable: `OPENAI_API_KEY`
- Config file editing

## Examples

```bash
# Configure with OpenRouter
policyascode config --api-key sk-or-... --base-url https://openrouter.ai/api/v1 --model openai/gpt-4o-mini

# Extract rules with custom prompt
policyascode extract --extraction-prompt "Extract compliance rules..." -o rules.json policy.md

# Consolidate with output
policyascode consolidate -i rules.json -o clean.json

# Validate with results saved to file
policyascode validate -r clean.json -o results.json document.md

# Show current config
policyascode config --show
```

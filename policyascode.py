import argparse, json, os, sys, base64, mimetypes, logging
from pathlib import Path
from typing import Dict, List, Any
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("policyascode")

CONFIG_DIR = Path.home() / ".policyascode"
CONFIG_FILE = CONFIG_DIR / "config"
SCHEMA_FILE = Path(__file__).resolve().parent / "config.json"
PROMPT_FILE = Path(__file__).resolve().parent / "prompt.json"

def load_config() -> Dict[str, str]:
    cfg = {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini"
    }
    if CONFIG_FILE.exists():
        for line in CONFIG_FILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip().replace("POLICYASCODE_", "").lower()] = v.strip().strip('"')
    return cfg

def save_config(cfg: Dict[str, str]) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    CONFIG_FILE.write_text(
        f'# Policy as Code CLI Configuration\n'
        f'POLICYASCODE_API_KEY="{cfg["api_key"]}"\n'
        f'POLICYASCODE_BASE_URL="{cfg["base_url"]}"\n'
        f'POLICYASCODE_MODEL="{cfg["model"]}"\n'
    )
    logger.info(f"Configuration saved to {CONFIG_FILE}")

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

SCHEMAS = load_json(SCHEMA_FILE).get("schemas", {})
PROMPTS = load_json(PROMPT_FILE)

def get_file_content(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    mime, _ = mimetypes.guess_type(path)
    if mime == "application/pdf":
        return {
            "type": "input_file",
            "filename": p.name,
            "file_data": f"data:application/pdf;base64,{base64.b64encode(p.read_bytes()).decode()}"
        }
    return {
        "type": "input_text",
        "text": f"# {p.name}\n\n{p.read_text(encoding='utf-8')}"
    }

def call_llm(cfg: Dict[str, str], instructions: str, content: Any, schema: Dict) -> Dict:
    if not cfg.get("api_key"):
        raise ValueError("API key not configured. Run: policyascode config --api-key YOUR_KEY")
    text = content.get("text", content.get("file_data", "")) if isinstance(content, dict) else str(content)
    safe_text = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")
    
    payload = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "system",
                "content": f"{instructions}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, separators=(',', ':'))}"
            },
            {
                "role": "user",
                "content": safe_text
            }
        ],
        "response_format": {"type": "json_object"},
        "stream": False
    }
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "policyascode-cli",
        "X-Title": "Policy as Code CLI"
    }
    
    logger.info(f"Calling model {cfg['model']} at {cfg['base_url']}")
    r = requests.post(f"{cfg['base_url']}/chat/completions", headers=headers, json=payload, timeout=60)
    
    if r.status_code != 200:
        raise RuntimeError(f"API Error {r.status_code}: {r.text}")
    result = r.json()
    if "error" in result:
        raise RuntimeError(result["error"].get("message", "Unknown API error"))
    
    try:
        return json.loads(result["choices"][0]["message"]["content"])
    except (KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Invalid JSON returned: {e}")

def extract_rules(cfg: Dict[str, str], files: List[str], out: str, custom_prompt: str = None) -> None:
    prompt = custom_prompt or PROMPTS.get("extract")
    all_rules = []
    idx = 0
    
    for f in files:
        try:
            logger.info(f"Processing {f}")
            content = get_file_content(f)
            fname = Path(f).name
            rules = call_llm(cfg, prompt, content, SCHEMAS["rules"]).get("rules", [])
            
            for i, r in enumerate(rules):
                r.update({"id": f"rule-{idx+i}", "source_file": fname})
                for s in r.get("sources", []):
                    s["file"] = fname
            
            all_rules.extend(rules)
            idx += len(rules)
            logger.info(f"Extracted {len(rules)} rules from {f}")
        except Exception as e:
            logger.error(f"{f}: {e}")
    
    Path(out).write_text(json.dumps({"rules": all_rules}, indent=2))
    logger.info(f"Saved {len(all_rules)} total rules to {out}")

def consolidate_rules(cfg: Dict[str, str], inp: str, out: str, custom_prompt: str = None) -> None:
    rules = load_json(Path(inp)).get("rules", [])
    if not rules:
        return logger.error("No rules found")
    
    prompt = custom_prompt or PROMPTS.get("consolidate")
    edits = call_llm(cfg, prompt, {"type": "input_text", "text": json.dumps(rules)}, SCHEMAS["edits"]).get("edits", [])
    
    if not edits:
        logger.info("No consolidation edits suggested")
        Path(out).write_text(json.dumps({"rules": rules}, indent=2))
        return
    
    lookup = {r["id"]: r for r in rules}
    to_delete = set()
    merged = []
    
    for e in edits:
        if e["edit"] in {"delete", "merge"}:
            to_delete.update(e["ids"])
        if e["edit"] == "merge":
            merged_sources = [s for rid in e["ids"] if rid in lookup for s in lookup[rid].get("sources", [])]
            src_files = list(set(lookup[rid].get("source_file", "") for rid in e["ids"] if rid in lookup))
            merged.append({
                "id": f"rule-merged-{'-'.join(e['ids'])}",
                "title": e["title"],
                "body": e["body"],
                "priority": e["priority"],
                "rationale": e["rationale"],
                "source_file": ", ".join(src_files) or "merged",
                "sources": merged_sources
            })
    
    final_rules = [r for r in rules if r["id"] not in to_delete] + merged
    Path(out).write_text(json.dumps({"rules": final_rules}, indent=2))
    logger.info(f"Applied {len(edits)} edits -> {len(final_rules)} rules saved to {out}")

def validate_documents(cfg: Dict[str, str], rules_file: str, docs: List[str], out: str = None, custom_prompt: str = None) -> None:
    rules = load_json(Path(rules_file)).get("rules", [])
    if not rules:
        return logger.error("No rules found")
    
    validations = []
    prompt_base = custom_prompt or PROMPTS.get("validate")
    
    for f in docs:
        fname = Path(f).name
        applicable = [r for r in rules if fname in r.get("source_file", "")]
        if not applicable:
            logger.warning(f"No applicable rules for {fname}")
            continue
        
        logger.info(f"Validating {fname} with {len(applicable)} rules")
        try:
            content = get_file_content(f)
            result = call_llm(cfg, f"{prompt_base}\n\nRules:\n{json.dumps(applicable)}", content, SCHEMAS["validation"])
            for v in result.get("validations", []):
                v["file"] = fname
            validations.extend(result.get("validations", []))
        except Exception as e:
            logger.error(f"{f}: {e}")
    
    if out:
        Path(out).write_text(json.dumps({"validations": validations}, indent=2))
        logger.info(f"Validation results saved to {out}")
    else:
        for v in validations:
            status = {"pass": "✅", "fail": "❌", "n/a": "⚪", "unknown": "❓"}.get(v["result"], "?")
            print(f"{status} {v['file']} :: {v['id']} -> {v['result'].upper()} - {v['reason']}")

def main() -> None:
    p = argparse.ArgumentParser(description="Policy as Code CLI")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--api-key")
    sub = p.add_subparsers(dest="cmd")

    e = sub.add_parser("extract", help="Extract rules from documents")
    e.add_argument("-o", "--output", required=True)
    e.add_argument("--extraction-prompt")
    e.add_argument("files", nargs="+")

    c = sub.add_parser("consolidate", help="Consolidate rules")
    c.add_argument("-i", "--input", required=True)
    c.add_argument("-o", "--output", required=True)
    c.add_argument("--consolidation-prompt")

    v = sub.add_parser("validate", help="Validate documents against rules")
    v.add_argument("-r", "--rules", required=True)
    v.add_argument("-o", "--output")
    v.add_argument("--validation-prompt")
    v.add_argument("files", nargs="+")

    conf = sub.add_parser("config", help="Configure API settings")
    conf.add_argument("--api-key")
    conf.add_argument("--base-url")
    conf.add_argument("--model")
    conf.add_argument("--show", action="store_true")

    args = p.parse_args()
    if not args.cmd:
        return p.print_help()

    cfg = load_config()
    if args.model:
        cfg["model"] = args.model
    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.api_key:
        cfg["api_key"] = args.api_key

    try:
        if args.cmd == "config":
            if args.show:
                print(f"Current configuration:\n  API Key: {cfg['api_key'][:10]}..." if cfg["api_key"] else "  API Key: Not set")
                print(f"  Base URL: {cfg['base_url']}\n  Model: {cfg['model']}\n  Config file: {CONFIG_FILE}")
            else:
                save_config(cfg)
        elif args.cmd == "extract":
            extract_rules(cfg, args.files, args.output, args.extraction_prompt)
        elif args.cmd == "consolidate":
            consolidate_rules(cfg, args.input, args.output, args.consolidation_prompt)
        elif args.cmd == "validate":
            validate_documents(cfg, args.rules, args.files, args.output, args.validation_prompt)
    except Exception as e:
        logger.error(e)
        sys.exit(1)

if __name__ == "__main__":
    main()

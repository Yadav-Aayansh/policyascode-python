import argparse, json, os, sys, base64, mimetypes, requests
from pathlib import Path
from typing import Dict, List, Any

class PolicyAsCode:
    def __init__(self):
        self.config_dir = Path.home() / '.policyascode'
        self.config_file = self.config_dir / 'config'
        self.config = self.load_config()
        self.schemas = self.load_schemas()
        
    def load_config(self) -> Dict[str, str]:
        """Load configuration from file or environment"""
        config = {"api_key": os.getenv('OPENAI_API_KEY', ''), "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o-mini"}
        if self.config_file.exists():
            for line in self.config_file.read_text().splitlines():
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip().replace('POLICYASCODE_', '').lower()
                    config[key] = value.strip().strip('"')
        return config
    
    def save_config(self):
        """Save configuration to file"""
        self.config_dir.mkdir(exist_ok=True)
        content = f"""# Policy as Code CLI Configuration\nPOLICYASCODE_API_KEY="{self.config['api_key']}"\nPOLICYASCODE_BASE_URL="{self.config['base_url']}"\nPOLICYASCODE_MODEL="{self.config['model']}"\n"""
        self.config_file.write_text(content)
        self.log_success(f"Configuration saved to {self.config_file}")
    
    def load_schemas(self) -> Dict[str, Any]:
        """Load JSON schemas from config.json"""
        config_path = Path(__file__).parent / 'config.json'
        if not config_path.exists(): config_path = Path('config.json')
        if config_path.exists(): return json.loads(config_path.read_text())['schemas']
        return {}
    
    def log_info(self, msg): print(f"\033[0;34m[INFO]\033[0m {msg}", file=sys.stderr)
    def log_success(self, msg): print(f"\033[0;32m[SUCCESS]\033[0m {msg}", file=sys.stderr)
    def log_warn(self, msg): print(f"\033[1;33m[WARN]\033[0m {msg}", file=sys.stderr)
    def log_error(self, msg): print(f"\033[0;31m[ERROR]\033[0m {msg}", file=sys.stderr)
    
    def get_file_content(self, filepath: str) -> Dict[str, Any]:
        """Get file content based on type"""
        path = Path(filepath)
        if not path.exists(): raise FileNotFoundError(f"File not found: {filepath}")
        mime_type, _ = mimetypes.guess_type(filepath)
        filename = path.name
        if mime_type == 'application/pdf':
            with open(filepath, 'rb') as f: data = base64.b64encode(f.read()).decode()
            return {"type": "input_file", "filename": filename, "file_data": f"data:application/pdf;base64,{data}"}
        else:
            content = path.read_text(encoding='utf-8')
            return {"type": "input_text", "text": f"# {filename}\n\n{content}"}
    
    def call_llm(self, instructions: str, content: Any, schema: Dict) -> Dict:
        """Make API call to LLM"""
        if not self.config.get('api_key'): raise ValueError("API key not configured. Run: policyascode config --api-key YOUR_KEY")
        if isinstance(content, dict): text_content = content.get('text', content.get('file_data', ''))
        else: text_content = str(content)
        text_content = ''.join(c for c in text_content if ord(c) >= 32 or c in '\n\t')
        enhanced_instructions = f"{instructions}\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema, separators=(',', ':'))}"
        headers = {"Authorization": f"Bearer {self.config['api_key']}", "Content-Type": "application/json", "HTTP-Referer": "policyascode-cli", "X-Title": "Policy as Code CLI"}
        payload = {"model": self.config['model'], "messages": [{"role": "system", "content": enhanced_instructions}, {"role": "user", "content": text_content}], "response_format": {"type": "json_object"}, "stream": False}
        self.log_info(f"Making API call to {self.config['base_url']}/chat/completions...")
        self.log_info(f"Using model: {self.config['model']}")
        response = requests.post(f"{self.config['base_url']}/chat/completions", headers=headers, json=payload, timeout=60)
        if response.status_code != 200: 
            error_msg = f"API Error {response.status_code}: {response.text}"
            self.log_error(error_msg)
            raise Exception(error_msg)
        result = response.json()
        if 'error' in result:
            error_msg = f"API Error: {result['error'].get('message', 'Unknown error')}"
            self.log_error(error_msg)
            raise Exception(error_msg)
        content = result['choices'][0]['message']['content']
        if not content: raise Exception("No content returned from API")
        content = ''.join(c for c in content if ord(c) >= 32 or c in '\n\t')
        try: return json.loads(content)
        except json.JSONDecodeError as e:
            self.log_error(f"Invalid JSON returned: {e}")
            raise Exception(f"Invalid JSON returned from API: {e}")
    
    def extract_rules(self, files: List[str], output: str, extraction_prompt: str = None):
        """Extract rules from documents"""
        if not extraction_prompt: extraction_prompt = """Extract atomic, testable rules from this policy document.
Keep each rule minimal.
Write for an LLM to apply it unambiguously.
Always include concise rationale and quotes.
Each rule should be specific to this document and its context."""
        self.log_info(f"Extracting rules from {len(files)} files...")
        all_rules, rule_index = [], 0
        for filepath in files:
            try:
                self.log_info(f"Processing file: {filepath}")
                content = self.get_file_content(filepath)
                result = self.call_llm(extraction_prompt, content, self.schemas['rules'])
                rules = result.get('rules', [])
                filename = Path(filepath).name
                for i, rule in enumerate(rules):
                    rule.update({'id': f'rule-{rule_index + i}', 'source_file': filename})
                    for source in rule.get('sources', []): source['file'] = filename
                all_rules.extend(rules)
                rule_index += len(rules)
                self.log_success(f"Extracted {len(rules)} rules from {filepath}")
            except Exception as e: self.log_error(f"Error processing {filepath}: {e}")
        Path(output).write_text(json.dumps({"rules": all_rules}, indent=2))
        self.log_success(f"Extracted {len(all_rules)} total rules to {output}")
    
    def consolidate_rules(self, input_file: str, output_file: str, consolidation_prompt: str = None):
        """Consolidate rules by removing duplicates and merging similar ones"""
        if not consolidation_prompt: consolidation_prompt = "Suggest deletes and merges to remove duplicates and generalize rules where appropriate."
        self.log_info(f"Loading rules from {input_file}")
        data = json.loads(Path(input_file).read_text())
        rules = data.get('rules', [])
        if not rules: 
            self.log_error("No rules found in input file")
            return
        self.log_info("Consolidating rules...")
        content = {"type": "input_text", "text": json.dumps(rules)}
        result = self.call_llm(consolidation_prompt, content, self.schemas['edits'])
        edits = result.get('edits', [])
        if not edits:
            self.log_info("No consolidation edits suggested")
            Path(output_file).write_text(json.dumps(data, indent=2))
            return
        rule_lookup = {rule['id']: rule for rule in rules}
        to_delete = set()
        merged_rules = []
        for edit in edits:
            if edit['edit'] in ['delete', 'merge']: to_delete.update(edit['ids'])
            if edit['edit'] == 'merge':
                merged_sources = []
                source_files = []
                for rule_id in edit['ids']:
                    if rule_id in rule_lookup:
                        merged_sources.extend(rule_lookup[rule_id].get('sources', []))
                        source_files.append(rule_lookup[rule_id].get('source_file', ''))
                merged_rule = {'id': f"rule-merged-{'-'.join(edit['ids'])}", 'title': edit['title'], 'body': edit['body'], 'priority': edit['priority'], 'rationale': edit['rationale'], 'source_file': ', '.join(set(f for f in source_files if f)) or 'merged', 'sources': merged_sources}
                merged_rules.append(merged_rule)
        final_rules = [rule for rule in rules if rule['id'] not in to_delete] + merged_rules
        Path(output_file).write_text(json.dumps({"rules": final_rules}, indent=2))
        self.log_success(f"Applied {len(edits)} consolidation edits, resulting in {len(final_rules)} rules saved to {output_file}")
    
    def validate_documents(self, rules_file: str, documents: List[str], output_file: str = None, validation_prompt: str = None):
        """Validate documents against rules"""
        if not validation_prompt: validation_prompt = """Validate the provided document against the given rules that originated from this same document.

For each rule, determine if the document passes, fails, is not applicable, or if it's unknown/unclear.

Return a validation result for each rule with:
- id: the rule identifier
- result: "pass" if the document complies, "fail" if it violates the rule, "n/a" if the rule is not applicable to this document, "unknown" if unclear or needs human review
- reason: brief explanation of why it passes, fails, is not applicable, or is unknown

Be specific and cite relevant parts of the document in your reasoning.

Only validate rules that were originally extracted from this document or are applicable to it."""
        rules_data = json.loads(Path(rules_file).read_text())
        rules = rules_data.get('rules', [])
        if not rules: 
            self.log_error("No rules found in rules file")
            return
        all_validations = []
        for filepath in documents:
            self.log_info(f"Validating file: {filepath}")
            filename = Path(filepath).name
            applicable_rules = [rule for rule in rules if (rule.get('source_file') == filename or filename in rule.get('source_file', '') or any(filename in sf for sf in rule.get('source_files', [])))]
            if not applicable_rules:
                self.log_warn(f"No applicable rules found for {filepath}")
                continue
            self.log_info(f"Found {len(applicable_rules)} applicable rules for {filepath}")
            try:
                content = self.get_file_content(filepath)
                full_prompt = f"{validation_prompt}\n\nRules to validate against:\n{json.dumps(applicable_rules)}"
                result = self.call_llm(full_prompt, content, self.schemas['validation'])
                validations = result.get('validations', [])
                for validation in validations: validation['file'] = filename
                all_validations.extend(validations)
                self.log_success(f"Validated {len(validations)} applicable rules against {filepath}")
            except Exception as e: self.log_error(f"Failed to validate {filepath}: {e}")
        if output_file:
            Path(output_file).write_text(json.dumps({"validations": all_validations}, indent=2))
            self.log_success(f"Validation results saved to {output_file}")
        else:
            by_file = {}
            for v in all_validations:
                file = v['file']
                if file not in by_file: by_file[file] = []
                by_file[file].append(v)
            for file, validations in by_file.items():
                print(f"\n=== Validation Results for {file} ===")
                for v in validations:
                    status = {'pass': '✅', 'fail': '❌', 'n/a': '⚪', 'unknown': '❓'}.get(v['result'], '?')
                    print(f"{status} {v['id']}: {v['result'].upper()} - {v['reason']}")
                print(f"  Total: {len(validations)} rules validated")

def main():
    parser = argparse.ArgumentParser(description="Policy as Code CLI")
    parser.add_argument('--model', help='LLM model to use')
    parser.add_argument('--base-url', help='API base URL')
    parser.add_argument('--api-key', help='API key')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    extract_parser = subparsers.add_parser('extract', help='Extract rules from documents')
    extract_parser.add_argument('-o', '--output', required=True, help='Output JSON file')
    extract_parser.add_argument('--extraction-prompt', help='Custom extraction prompt')
    extract_parser.add_argument('files', nargs='+', help='Input documents')
    consolidate_parser = subparsers.add_parser('consolidate', help='Consolidate rules')
    consolidate_parser.add_argument('-i', '--input', required=True, help='Input rules JSON file')
    consolidate_parser.add_argument('-o', '--output', required=True, help='Output consolidated JSON file')
    consolidate_parser.add_argument('--consolidation-prompt', help='Custom consolidation prompt')
    validate_parser = subparsers.add_parser('validate', help='Validate documents against rules')
    validate_parser.add_argument('-r', '--rules', required=True, help='Rules JSON file')
    validate_parser.add_argument('-o', '--output', help='Output JSON file for validation results')
    validate_parser.add_argument('--validation-prompt', help='Custom validation prompt')
    validate_parser.add_argument('files', nargs='+', help='Documents to validate')
    config_parser = subparsers.add_parser('config', help='Configure API settings')
    config_parser.add_argument('--api-key', help='Set API key')
    config_parser.add_argument('--base-url', help='Set API base URL')
    config_parser.add_argument('--model', help='Set default model')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    pac = PolicyAsCode()
    if args.model: pac.config['model'] = args.model
    if args.base_url: pac.config['base_url'] = args.base_url
    if args.api_key: pac.config['api_key'] = args.api_key
    try:
        if args.command == 'config':
            if args.api_key: pac.config['api_key'] = args.api_key
            if args.base_url: pac.config['base_url'] = args.base_url
            if args.model: pac.config['model'] = args.model
            if args.show:
                print("Current configuration:")
                print(f"  API Key: {pac.config['api_key'][:10]}..." if pac.config['api_key'] else "  API Key: Not set")
                print(f"  Base URL: {pac.config['base_url']}")
                print(f"  Model: {pac.config['model']}")
                print(f"  Config file: {pac.config_file}")
            else: pac.save_config()
        elif args.command == 'extract':
            if not pac.config.get('api_key'):
                pac.log_error("API key is required. Set it using: policyascode config --api-key <your-key>")
                sys.exit(1)
            pac.extract_rules(args.files, args.output, args.extraction_prompt)
        elif args.command == 'consolidate':
            if not pac.config.get('api_key'):
                pac.log_error("API key is required. Set it using: policyascode config --api-key <your-key>")
                sys.exit(1)
            pac.consolidate_rules(args.input, args.output, args.consolidation_prompt)
        elif args.command == 'validate':
            if not pac.config.get('api_key'):
                pac.log_error("API key is required. Set it using: policyascode config --api-key <your-key>")
                sys.exit(1)
            pac.validate_documents(args.rules, args.files, args.output, args.validation_prompt)
    except Exception as e:
        pac.log_error(str(e))
        sys.exit(1)

if __name__ == '__main__': main()

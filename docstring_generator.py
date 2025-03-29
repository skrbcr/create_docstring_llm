import os
import ast
import shutil
import re
import sys
import logging
from typing import Optional
import requests
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, style: str = "google",
                 llm: str = "openrouter", url: Optional[str] = None, model: Optional[str] = None):
        self.llm_type = llm.lower()
        if self.llm_type == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        else:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.style = style
        if not self.api_key:
            logger.error("LLM API key not found. Set OPENROUTER_API_KEY environment variable or use --api-key")
            sys.exit(1)
        
        # Configure API URL and default model based on llm type
        if self.llm_type == "openai":
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.model = model if model is not None else "gpt-3.5-turbo"
        elif self.llm_type == "ollama":
            if not url:
                logger.error("Ollama selected but no URL provided. Use --url option.")
                sys.exit(1)
            self.api_url = url
            self.model = model if model is not None else "default_ollama_model"
        else:  # Default to openrouter
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model if model is not None else "deepseek/deepseek-chat-v3-0324:free"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str) -> str:
        """Generate docstring using LLM API with retry for 429 errors"""
        import time
        if self.llm_type == "openai":
            client = OpenAI()
            attempt = 0
            while attempt < 3:
                try:
                    response = client.responses.create(
                        model=self.model,
                        input=f"Generate a {self.style}-style docstring for the following Python function. Return only the generated docstring including both triple quotes. Ensure the output starts and ends with triple quotes. Function code:\n\n{prompt}",
                    )
                    content = response.output_text
                    return content
                except Exception as e:
                    if "429" in str(e):
                        logger.warning("Received 429 error from OpenAI API. Waiting for 30 seconds before retrying...")
                        time.sleep(30)
                        attempt += 1
                    else:
                        logger.error(f"OpenAI API error: {e}")
                        return ""
            logger.error("Failed to generate docstring after 3 attempts due to rate limiting.")
            return ""
        else:
            payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": f"Generate a {self.style}-style docstring for the following Python function. Return only the generated docstring including both triple quotes (\"\"\"). Make sure to wrap the output with \"\"\" at both the beginning and the end. function:\n\n{prompt}"
                }],
                "temperature": 0.2,
                "max_tokens": 500
            }
            attempt = 0
            while attempt < 3:
                try:
                    response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
                    if response.status_code == 429:
                        logger.warning("Rate limit exceeded (429). Waiting for 30 seconds before retrying...")
                        time.sleep(30)
                        attempt += 1
                        continue
                    response.raise_for_status()
                    print(prompt)
                    print(response.json())
                    content = response.json()["choices"][0]["message"]["content"]
                    return content
                except Exception as e:
                    if "429" in str(e):
                        logger.warning("Received 429 error from API. Waiting for 30 seconds before retrying...")
                        time.sleep(30)
                        attempt += 1
                        continue
                    logger.error(f"LLM API error: {e}")
                    return ""
            logger.error("Failed to generate docstring after 3 attempts due to rate limiting.")
            return ""

class DocstringGenerator:
    def __init__(self, input_dir: str, output_dir: str, api_key: Optional[str] = None,
                 style: str = "google", overwrite: bool = False, llm: str = "openrouter",
                 url: Optional[str] = None, model: Optional[str] = None, num_threads: int = 4):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.llm = LLMClient(api_key, style, llm, url, model)
        self.overwrite = overwrite
        self.num_threads = num_threads
        self.processed_count = 0
        self.skipped_count = 0

    def process_directory(self):
        """Process all Python files in the input directory recursively"""
        if not os.path.isdir(self.input_dir):
            logger.error(f"Input directory not found: {self.input_dir}")
            return

        logger.info(f"Processing directory: {self.input_dir}")
        logger.info(f"Output will be saved to: {self.output_dir}")

        if os.path.exists(self.output_dir):
            logger.info("Cleaning existing output directory")
            shutil.rmtree(self.output_dir)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".py"):
                    input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, self.input_dir)
                    output_path = os.path.join(self.output_dir, relative_path, file)
                    
                    logger.info(f"Processing file: {input_path}")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    self.process_file(input_path, output_path)
                    self.processed_count += 1
        
        logger.info(f"Processing complete. Processed: {self.processed_count}, Skipped: {self.skipped_count}")

    def process_file(self, input_path: str, output_path: str):
        """Process a single Python file while preserving original formatting except for docstring updates."""
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            new_source = update_source_with_docstrings(source, self.llm, self.overwrite)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(new_source)
                
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            self.skipped_count += 1

class ASTProcessor(ast.NodeTransformer):
    def __init__(self, llm: LLMClient, overwrite: bool = False):
        self.llm = llm
        self.overwrite = overwrite
        super().__init__()
        self.current_class = None

    def visit_FunctionDef(self, node):
        # Skip if already updated or should be skipped
        if not getattr(node, "_docstring_updated", False) and not self._should_skip(node):
            node = self._generate_docstring(node)
        return self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        # Update class docstring if not processed and not skipped
        if not getattr(node, "_docstring_updated", False) and not self._should_skip(node):
            node = self._generate_docstring(node)
        # Set current class for processing methods
        self.current_class = node.name
        node = self.generic_visit(node)
        self.current_class = None
        return node

    def _should_skip(self, node) -> bool:
        """Check if node should be skipped.
        
        Currently, only nodes inside a __main__ block are skipped.
        """
        parent = getattr(node, "parent", None)
        while parent:
            if (isinstance(parent, ast.If) and 
                isinstance(parent.test, ast.Compare) and
                isinstance(parent.test.left, ast.Name) and
                parent.test.left.id == "__name__"):
                return True
            parent = getattr(parent, "parent", None)
        return False

    def _clean_node_body(self, node):
        """Remove implementation code while preserving docstrings"""
        # Remove all non-docstring nodes
        node.body = [
            n for n in node.body 
            if isinstance(n, ast.Expr) and 
            isinstance(n.value, (ast.Str, ast.Constant)) and
            (isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))
        ]

    def _generate_docstring(self, node):
        """
        Generate a new docstring using LLM output and update the node's docstring.
        """
        try:
            # Skip if node already has a docstring and overwrite is False
            if not self.overwrite and ast.get_docstring(node) is not None:
                return node
                
            original_docstring = ast.get_docstring(node)
            if original_docstring:
                # Temporarily remove existing docstring
                node.body = node.body[1:] if isinstance(node.body[0], ast.Expr) else node.body
                
            node_def = ast.unparse(node)
            response = self.llm.generate(node_def)
            
            if response:
                # Create docstring node from response
                docstring_node = ast.Expr(value=ast.Constant(value=response.strip('"')))
                # Insert docstring at beginning of node body
                node.body.insert(0, docstring_node)
                node._docstring_updated = True
        except Exception as e:
            logger.warning(f"Failed to generate docstring for {node.name}: {e}")
        return node

def update_source_with_docstrings(source: str, llm, overwrite: bool) -> str:
    """
    Update the docstrings in the source code using the provided LLM.
    This function preserves all code formatting except for updates to function/method docstrings.
    It processes top-level functions, methods, and class definitions (excluding those in a __main__ block)
    and inserts or replaces the docstring based on the 'overwrite' flag.
    Docstring generation is performed concurrently.
    """
    import re
    from concurrent.futures import ThreadPoolExecutor
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    parent_map = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent

    def is_inside_main(node):
        current = node
        while current in parent_map:
            parent = parent_map[current]
            if isinstance(parent, ast.If):
                test = parent.test
                if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"):
                    return True
            current = parent
        return False

    targets = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if is_inside_main(node):
                continue
            if not overwrite and ast.get_docstring(node, clean=False) is not None:
                continue
            targets.append(node)
    
    targets.sort(key=lambda n: n.lineno, reverse=True)

    results = {}
    with ThreadPoolExecutor() as executor:
        future_map = {node: executor.submit(llm.generate, "".join(lines[node.lineno - 1:node.end_lineno])) for node in targets}
        for node, future in future_map.items():
            results[node] = future.result()

    for node in targets:
        new_doc = results.get(node)
        if not new_doc:
            continue
        has_doc = False
        if node.body and isinstance(node.body[0], ast.Expr):
            first_expr = node.body[0]
            if (hasattr(first_expr, "value") and isinstance(first_expr.value, (ast.Constant, ast.Constant)) and isinstance(ast.get_docstring(node, clean=False), str)):
                has_doc = True
                doc_start = first_expr.lineno - 1
                doc_end = first_expr.end_lineno
        indent = ""
        if node.body:
            line_str = lines[node.body[0].lineno - 1]
            indent_match = re.match(r"(\s*)", line_str)
            if indent_match:
                indent = indent_match.group(1)
        else:
            def_line = lines[node.lineno - 1]
            indent_match = re.match(r"(\s*)", def_line)
            indent = (indent_match.group(1) + "    ") if indent_match else "    "
        doc_lines = new_doc.splitlines()
        formatted_doc = "\n".join(indent + line if line.strip() != "" else line for line in doc_lines) + "\n"
        if has_doc:
            lines[doc_start:doc_end] = [formatted_doc]
        else:
            insertion_index = node.lineno
            lines.insert(insertion_index, formatted_doc)
    return "".join(lines)

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate Python stub files with docstrings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing Python files to process"
    )
    parser.add_argument(
        "--api-key",
        help="LLM API key. For 'openai' llm type, overrides OPENAI_API_KEY. For others, overrides OPENROUTER_API_KEY."
    )
    parser.add_argument(
        "--style",
        choices=["google", "numpy", "rest"],
        default="google",
        help="Docstring style (google, numpy, rest)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing docstrings"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory where processed files will be written"
    )
    parser.add_argument(
        "--llm",
        choices=["openai", "openrouter", "ollama"],
        default="openrouter",
        help="LLM type to use. For 'openai', use OPENAI_API_KEY; for others, use OPENROUTER_API_KEY."
    )
    parser.add_argument(
        "--url",
        help="URL for the LLM server; required if using 'ollama', ignored for other llm types."
    )
    parser.add_argument(
        "--model",
        help="LLM model name. Defaults: openai: 'gpt-3.5-turbo'; openrouter: 'deepseek/deepseek-chat-v3-0324:free'; ollama: 'default_ollama_model'."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "-n", "--num-threads", type=int, default=4,
        help="Number of threads to use for parallel processing"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    generator = DocstringGenerator(
        args.input_dir,
        args.output_dir,
        api_key=args.api_key,
        style=args.style,
        overwrite=args.overwrite,
        llm=args.llm,
        url=args.url,
        model=args.model,
        num_threads=args.num_threads
    )
    generator.process_directory()

if __name__ == "__main__":
    main()

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
        """Generate docstring using LLM API"""
        if self.llm_type == "openai":
            client = OpenAI()

            response = client.responses.create(
                model=self.model,
                input=f"Generate a {self.style}-style docstring for the following Python function. Return only the generated docstring including both triple quotes. Ensure the output starts and ends with triple quotes. Function code:\n\n{prompt}",
            )
            try:
                content = response.output_text
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
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
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
                response.raise_for_status()
                print(prompt)
                print(response.json())
                content = response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.error(f"LLM API error: {e}")
                return ""
        try:
            # Extract just the docstring content
            docstring_match = re.search(r'^(""".*?""")', content, re.MULTILINE | re.DOTALL)
            return docstring_match.group(1) if docstring_match else ""
        except Exception as e:
            logger.error(f"Error extracting docstring: {e}")
            return ""

class DocstringGenerator:
    def __init__(self, input_dir: str, output_dir: str, api_key: Optional[str] = None,
                 style: str = "google", overwrite: bool = False, llm: str = "openrouter",
                 url: Optional[str] = None, model: Optional[str] = None):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.llm = LLMClient(api_key, style, llm, url, model)
        self.overwrite = overwrite
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
        """Process a single Python file"""
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            processor = ASTProcessor(self.llm, self.overwrite)
            processor.visit(tree)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(ast.unparse(tree))
                
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
        model=args.model
    )
    generator.process_directory()

if __name__ == "__main__":
    main()

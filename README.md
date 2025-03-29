# Python Docstring Generator Tool

A tool that automatically generates docstrings for Python code using LLM APIs. It processes Python files recursively and creates files with inserted docstrings while preserving the original codes.

## Overview

This tool scans a specified directory for Python files and generates docstrings for functions and class methods using an LLM. It supports multiple docstring styles (Google, NumPy, reStructuredText) and can be configured either to insert new docstrings or overwrite existing ones.

## Features

- Recursively processes Python files.
- Supports multiple docstring styles:
  - Google (default)
  - NumPy
  - reStructuredText
- Integrates with various LLM services:
  - OpenRouter API (default)
  - OpenAI API
  - ollama (local LLM)
- Option to overwrite existing docstrings.
- Preserves original function and class implementations while updating or inserting docstrings.
- Recreates the directory structure in the specified output directory.
- Parallel processing.

## Pre-requisites

- An LLM API key:
  - For OpenRouter (default): provide via the `OPENROUTER_API_KEY` environment variable or the `--api-key` option.
  - For OpenAI: provide via the `OPENAI_API_KEY` environment variable or the `--api-key` option when using `--llm openai`.
- [uv](https://github.com/astral-sh/uv) (recommended Python package manager).

## Installation

1. Clone this repository.
2. Install dependencies:

```bash
uv sync
```

## Usage

The script takes an input directory and an output directory as positional arguments. It processes all `.py` files in the input directory (including in subdirectories) and writes the processed files with generated docstrings to the output directory, preserving the directory structure.

Basic command:

```bash
uv run docstring_generator.py [input_directory] [output_directory] [options]
```

### Options

- `--api-key`: LLM API key (overrides environment variable).
- `--style`: Docstring style. Choices: `google` (default), `numpy`, `rest`.
- `--overwrite`: Overwrite existing docstrings (default: skip).
- `--llm`: LLM service to use. Choices: `openai`, `openrouter` (default), `ollama`.
- `--url`: URL for the LLM server (required if using `ollama`).
- `--model`: LLM model name. Options and defaults:
  - OpenAI: `gpt-3.5-turbo` but **you should use more recent models**.
  - OpenRouter: `deepseek/deepseek-chat-v3-0324:free`
  - ollama: `default_ollama_model`
- `--num` or `-n`: Number of threads for parallel processing (default: 1).
- `--verbose` or `-v`: Enable detailed logging.

### Examples

1. Process the `input/` directory and output to `output/`:

```bash
uv run docstring_generator.py ./input ./output
```

2. Process with an explicit API key and verbose logging:

```bash
uv run docstring_generator.py ./input ./output --api-key sk-xxxxxxxx -v
```

3. Process with NumPy style docstrings and overwrite existing ones:

```bash
uv run docstring_generator.py ./input ./output --style numpy --overwrite
```

4. Full example with all options:

```bash
uv run docstring_generator.py ./input ./output --api-key sk-xxxxxxxx --style rest --overwrite --llm openrouter --model deepseek/deepseek-chat-v3-0324:free -v
```

## Output Structure

The tool creates output files in the specified output directory, maintaining the directory structure of the input. For example:

```
project/
├── input/                    # Input directory
│   ├── module1.py
│   └── subdir/
│       └── module2.py
└── output/                   # Output directory
    ├── module1.py          # With generated docstrings
    └── subdir/
        └── module2.py      # With generated docstrings
```

## Notes

- Only Python files (`.py`) are processed.
- The tool will overwrite files in the output directory if they exist.
- If docstring generation fails for a function or class, that element is skipped.
- Existing docstrings are preserved unless the `--overwrite` option is used.

## Contributing

Contributions are welcome!
This tool may contain bugs or potential improvements.
Feel free to open an issue or submit a pull request ;)

## Acknowledgements

- Developed with [Roo-Code](https://github.com/RooVetGit/Roo-Code).
- Utilizes LLM services for generating docstrings.
- ChatGPT contributed in creating the requirements definition document (rdd.md).

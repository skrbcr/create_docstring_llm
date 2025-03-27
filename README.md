# Python Docstring Generator Tool

A tool that automatically generates Google-style docstrings for Python code using OpenRouter's API. Processes Python files recursively and creates stub files with docstrings in a `docstring/` directory while preserving the original function/class structure.

## Features

- Processes Python files recursively
- Preserves function/class structure
- Generates Google-style docstrings
- Handles LLM response formatting automatically
- Fallback behavior when docstring generation fails
- Preserves original function bodies while only updating docstrings
- Creates output directory structure matching input

## Pre-requisites

- [OpenRouter](https://openrouter.ai/) API key (you can get one for free)
  - Can be provided via:
    - `OPENROUTER_API_KEY` environment variable
    - `--api-key` command line argument
- [uv](https://github.com/astral-sh/uv) (recommended Python package manager)

## Installation

1. Clone this repository
2. Install dependencies:

```bash
uv sync
```

## Usage

Basic command:

```bash
uv run docstring_generator.py [input_directory] [options]
```

### Options

- `--api-key`: OpenRouter API key (overrides environment variable)
- `--verbose` or `-v`: Enable detailed logging

### Examples

1. Process `src/` directory using environment variable for API key:

```bash
uv run docstring_generator.py ./src
```

2. Process `src/` with explicit API key and verbose output:

```bash
uv run docstring_generator.py ./src --api-key sk-xxxxxxxx -v
```

## Output Structure

The tool creates a parallel directory structure under `docstring/`:

```
project/
├── src/                    # Input directory
│   ├── module1.py
│   └── subdir/
│       └── module2.py
└── docstring/              # Output directory
    ├── src/
    │   ├── module1.py      # With generated docstrings
    │   └── subdir/
    │       └── module2.py  # With generated docstrings
```

## Notes

- The tool will overwrite existing files in the `docstring/` directory
- Only Python files (`.py`) are processed
- Original function/class implementations are preserved - only docstrings are updated
- If docstring generation fails for a function/class, it will be skipped

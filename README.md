# Python Docstring Generator Tool

A tool that automatically generates Google-style docstrings for Python code using OpenRouter's API. Processes Python files recursively and creates stub files with docstrings in a `docstring/` directory while preserving the original function/class structure.

## Features

- Processes Python files recursively
- Preserves function/class structure
- Generates docstrings in multiple styles (Google, NumPy, reStructuredText)
- Handles LLM response formatting automatically
- Option to overwrite existing docstrings
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
- `--style`: Docstring style [google|numpy|rest] (default: google)
- `--overwrite`: Overwrite existing docstrings (default: skip)
- `--output-dir`, `-o`: Output directory (required)
- `--verbose` or `-v`: Enable detailed logging

### Examples

1. Process `input/` directory with output to `output/`:

```bash
uv run docstring_generator.py ./input -o ./output
```

2. Process with explicit API key and verbose output:

```bash
uv run docstring_generator.py ./input -o ./output --api-key sk-xxxxxxxx -v
```

3. Process with NumPy style docstrings and overwrite existing ones:

```bash
uv run docstring_generator.py ./input -o ./output --style numpy --overwrite
```

4. Full example with all options:

```bash
uv run docstring_generator.py ./input \
    -o ./output \
    --api-key sk-xxxxxxxx \
    --style rest \
    --overwrite \
    -v
```

## Output Structure

The tool creates output files in the specified directory (`-o/--output-dir`), maintaining the input directory structure:

Example output structure:

```
project/
├── input/                    # Input directory
│   ├── module1.py
│   └── subdir/
│       └── module2.py
└── output/                   # Output directory (specified by -o/--output-dir)
    ├── module1.py          # With generated docstrings
    └── subdir/
        └── module2.py      # With generated docstrings
```

## Notes

- The tool will overwrite existing files in the `docstring/` directory
- Only Python files (`.py`) are processed
- Original function/class implementations are preserved - only docstrings are updated
- If docstring generation fails for a function/class, it will be skipped

## Acknowledgements

- This tool is developed with [Roo-Code](https://github.com/RooVetGit/Roo-Code).
    - `DeepSeek-V3-0324 model` is used for code generation.
- ChatGPT helps me with creating requiments definition document (`rdd.md`).

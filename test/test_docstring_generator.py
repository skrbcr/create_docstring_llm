import os
import subprocess
import pytest

def read_file(filepath):
    with open(filepath, "r") as f:
        return f.read()

def run_generator(input_dir, output_dir, extra_args=None):
    command = ["uv", "run", "docstring_generator.py", input_dir, output_dir, "--llm", "openai", "--model", "gpt-4o-2024-11-20"]
    if extra_args:
        command.extend(extra_args)
    # Run the command and wait for it to complete
    subprocess.check_call(command)

def test_no_overwrite():
    # Setup input and output directories
    input_dir = os.path.join("test", "testcases")
    output_dir = os.path.join("test", "output_no_overwrite")
    run_generator(input_dir, output_dir)
    
    sample1_path = os.path.join(output_dir, "sample1.py")
    sample2_path = os.path.join(output_dir, "sample2.py")
    
    sample1_content = read_file(sample1_path)
    sample2_content = read_file(sample2_path)
    
    # Functions with existing docstrings should remain unchanged.
    assert '"""Existing docstring."""' in sample1_content
    assert '"""Existing docstring."""' in sample2_content
    
    # Functions/methods without an existing docstring should get a new one.
    # We assume the new docstring is "Generated docstring." inserted just after the def line.
    assert "Generated docstring." in sample1_content
    assert "Generated docstring." in sample2_content

def test_overwrite():
    # Setup input and output directories for overwrite mode
    input_dir = os.path.join("test", "testcases")
    output_dir = os.path.join("test", "output_overwrite")
    run_generator(input_dir, output_dir, extra_args=["--overwrite"])
    
    sample1_path = os.path.join(output_dir, "sample1.py")
    sample2_path = os.path.join(output_dir, "sample2.py")
    
    sample1_content = read_file(sample1_path)
    sample2_content = read_file(sample2_path)
    
    # In overwrite mode, all functions and methods should have "Generated docstring." regardless of prior content.
    # Check sample1 for both functions.
    # func_with_docstring should have been overwritten.
    assert "Generated docstring." in sample1_content
    # Check sample2 for both methods.
    assert "Generated docstring." in sample2_content

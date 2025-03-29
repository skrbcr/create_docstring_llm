import os
import re
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

def extract_docstring(content, func_name):
    # Regex to extract a triple-quoted docstring from a function definition.
    pattern = r'def\s+' + re.escape(func_name) + r'\s*\([^)]*\):\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1) or match.group(2)
    return None

def test_no_overwrite():
    # Setup input and output directories
    input_dir = os.path.join("test", "testcases")
    output_dir = os.path.join("test", "output_no_overwrite")
    
    # Read original files to get expected existing docstrings.
    original_sample1 = read_file(os.path.join("test", "testcases", "sample1.py"))
    original_sample2 = read_file(os.path.join("test", "testcases", "sample2.py"))
    expected_doc_func = extract_docstring(original_sample1, "func_with_docstring")
    expected_doc_method = extract_docstring(original_sample2, "method_with_docstring")
    
    run_generator(input_dir, output_dir)
    
    sample1_path = os.path.join(output_dir, "sample1.py")
    sample2_path = os.path.join(output_dir, "sample2.py")
    
    sample1_content = read_file(sample1_path)
    sample2_content = read_file(sample2_path)
    
    # For sample1.py:
    # "func_with_docstring" should retain its original docstring exactly.
    docstring1 = extract_docstring(sample1_content, "func_with_docstring")
    assert docstring1 is not None, "func_with_docstring should have a docstring."
    assert docstring1.strip() == expected_doc_func.strip(), "Original docstring should be preserved for func_with_docstring."
    
    # "func_without_docstring" should now have a new non-empty docstring.
    docstring2 = extract_docstring(sample1_content, "func_without_docstring")
    assert docstring2 is not None, "func_without_docstring should now have a docstring."
    assert docstring2.strip() != "", "The inserted docstring for func_without_docstring should not be empty."
    
    # For sample2.py in class SampleClass:
    # "method_with_docstring" should retain its original docstring exactly.
    docstring3 = extract_docstring(sample2_content, "method_with_docstring")
    assert docstring3 is not None, "method_with_docstring should have a docstring."
    assert docstring3.strip() == expected_doc_method.strip(), "Original docstring should be preserved for method_with_docstring."
    
    # "method_without_docstring" should now have a new non-empty docstring.
    docstring4 = extract_docstring(sample2_content, "method_without_docstring")
    assert docstring4 is not None, "method_without_docstring should now have a docstring."
    assert docstring4.strip() != "", "The inserted docstring for method_without_docstring should not be empty."

def test_overwrite():
    # Setup input and output directories for overwrite mode.
    input_dir = os.path.join("test", "testcases")
    output_dir = os.path.join("test", "output_overwrite")
    run_generator(input_dir, output_dir, extra_args=["--overwrite"])
    
    sample1_path = os.path.join(output_dir, "sample1.py")
    sample2_path = os.path.join(output_dir, "sample2.py")
    
    sample1_content = read_file(sample1_path)
    sample2_content = read_file(sample2_path)
    
    # In overwrite mode, every function/method should have a docstring.
    docstring1 = extract_docstring(sample1_content, "func_with_docstring")
    assert docstring1 is not None, "In overwrite mode, func_with_docstring should have a docstring."
    
    docstring2 = extract_docstring(sample1_content, "func_without_docstring")
    assert docstring2 is not None, "In overwrite mode, func_without_docstring should have a docstring."
    
    docstring3 = extract_docstring(sample2_content, "method_with_docstring")
    assert docstring3 is not None, "In overwrite mode, method_with_docstring should have a docstring."
    
    docstring4 = extract_docstring(sample2_content, "method_without_docstring")
    assert docstring4 is not None, "In overwrite mode, method_without_docstring should have a docstring."

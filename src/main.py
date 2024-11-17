# This is the main directory for the ChipFlow AI exercise

import os
import sys
import ast
import pytest
import tempfile
import google.generativeai as genai
import subprocess
from typing import Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeGenerationAgent:
    def __init__(self, api_key: Optional[str] = None, max_iterations: int = 3):
        """
        Initialize the code generation agent.
        
        Args:
            api_key: LLM API key
            max_iterations: Maximum number of iteration attempts
        """
        genai.configure(api_key=os.environ["API_KEY"])
        #self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.max_iterations = max_iterations
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
    def generate_code(self, requirements: str) -> str:
        """Generate Python code based on requirements."""
        prompt = f"""Write a Python program that meets these requirements:
{requirements}

The code should be complete, well-documented, and follow Python best practices.
Only provide the code without any explanations."""

        try:
            response = self.model.generate_content(prompt)
            logger.info("Generated code!")
            return response.text
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise

    def generate_test(self, code: str, requirements: str) -> str:
        """Generate pytest test cases based on the code and requirements."""
        prompt = f"""Write pytest test cases for the following Python code and requirements:

Requirements:
{requirements}

Code:
{code}

Generate comprehensive test cases that cover main functionality and edge cases.
Only provide the test code without any explanations."""

        try:
            response = self.model.generate_content(prompt)
            logger.info("Generated tests!")
            return response.text
        except Exception as e:
            logger.error(f"Error generating tests: {str(e)}")
            raise

    def improve_code(self, code: str, test_output: str, requirements: str) -> str:
        """Generate improved code based on test failures."""
        prompt = f"""Improve the following Python code to fix the test failures:

Requirements:
{requirements}

Current Code:
{code}

Test Output:
{test_output}

Generate an improved version of the code that passes the tests.
Only provide the code without any explanations."""

        try:
            response = self.model.generate_content(prompt)
            logger.info("Improved code!")
            return response.text
        except Exception as e:
            logger.error(f"Error improving code: {str(e)}")
            raise

    def validate_syntax(self, code: str) -> bool:
        """Validate Python code syntax."""
        try:
            split_code = code.split("\n")
            for is_it_tilde in split_code:
                if "```" in is_it_tilde:
                    split_code.remove(is_it_tilde)
            split_code_join = "\n".join(split_code)
            ast.parse(split_code_join)
            return True
        except SyntaxError:
            return False

    def run_tests(self, code: str, test_code: str) -> Tuple[bool, str]:
        """Execute the test cases and return results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save code and test files
            code_path = os.path.join(tmpdir, "solution.py")
            test_path = os.path.join(tmpdir, "test_solution.py")
            
            with open(code_path, 'w') as f:
                f.write(code)
            with open(test_path, 'w') as f:
                f.write(test_code)

            # Run pytest
            process = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-v"],
                capture_output=True,
                text=True,
                cwd=tmpdir
            )
            
            tests_passed = process.returncode == 0
            return tests_passed, process.stdout + process.stderr

    def generate_and_test(self, requirements: str) -> Dict[str, str]:
        """
        Main method to generate code, tests, and iterate until success.
        
        Returns:
            Dict containing final code, tests, and test results
        """
        iteration = 0
        while iteration < self.max_iterations:
            logger.info(f"Starting iteration {iteration + 1}")
            
            # Generate initial code
            if iteration == 0:
                code = self.generate_code(requirements)
            
            # Validate syntax
            if not self.validate_syntax(code):
                logger.warning("Generated code has syntax errors. Retrying...")
                code = self.generate_code(requirements)
                continue
            
            # Generate tests
            test_code = self.generate_test(code, requirements)
            if not self.validate_syntax(test_code):
                logger.warning("Generated tests have syntax errors. Retrying...")
                continue
            
            # Run tests
            tests_passed, test_output = self.run_tests(code, test_code)
            
            if tests_passed:
                logger.info("All tests passed!")
                return {
                    "code": code,
                    "tests": test_code,
                    "test_output": test_output,
                    "iterations": iteration + 1
                }
            
            # If tests failed, try to improve the code
            logger.info("Tests failed. Attempting to improve code...")
            code = self.improve_code(code, test_output, requirements)
            iteration += 1
        
        logger.warning(f"Failed to generate working code after {self.max_iterations} iterations")
        return {
            "code": code,
            "tests": test_code,
            "test_output": test_output,
            "iterations": iteration
        }

def main():
    """CLI interface for the code generation agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate and test Python code from requirements')
    parser.add_argument('--requirements', '-r', required=True, help='Requirements for the Python program')
    parser.add_argument('--output-dir', '-o', default='generated', help='Output directory')
    parser.add_argument('--max-iterations', '-m', type=int, default=3, help='Maximum number of improvement iterations')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize and run agent
    agent = CodeGenerationAgent(max_iterations=args.max_iterations)
    result = agent.generate_and_test(args.requirements)
    
    # Save results
    with open(os.path.join(args.output_dir, 'solution.py'), 'w') as f:
        f.write(result['code'])
    with open(os.path.join(args.output_dir, 'test_solution.py'), 'w') as f:
        f.write(result['tests'])
    with open(os.path.join(args.output_dir, 'test_output.txt'), 'w') as f:
        f.write(result['test_output'])
    
    logger.info(f"Generated files saved to {args.output_dir}")
    logger.info(f"Completed in {result['iterations']} iterations")
    
if __name__ == "__main__":
    main()

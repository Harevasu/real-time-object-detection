"""
Test runner script for Real-Time Object Detection.
Run all tests with coverage report.
"""

import subprocess
import sys


def run_tests():
    """Run pytest with coverage."""
    print("=" * 60)
    print("  Running Unit Tests")
    print("=" * 60)
    
    # Run pytest with coverage
    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        'tests/',
        '-v',
        '--tb=short',
        '--cov=src',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov'
    ])
    
    return result.returncode


def run_tests_quick():
    """Run pytest without coverage (faster)."""
    print("=" * 60)
    print("  Running Quick Tests")
    print("=" * 60)
    
    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        'tests/',
        '-v',
        '--tb=short',
        '-x'  # Stop on first failure
    ])
    
    return result.returncode


def run_specific_test(test_name):
    """Run a specific test file or test."""
    print(f"Running: {test_name}")
    
    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        test_name,
        '-v',
        '--tb=long'
    ])
    
    return result.returncode


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            exit(run_tests_quick())
        else:
            exit(run_specific_test(sys.argv[1]))
    else:
        exit(run_tests())

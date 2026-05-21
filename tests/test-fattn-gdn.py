#!/usr/bin/env python3
"""
CMake test wrapper for flash attention in gated delta net.
Usage: ctest -R test-fattn-gdn -V
"""

import sys
import os
import subprocess

def main():
    """Run the Python integration tests for flash attention."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_script = os.path.join(test_dir, "python", "test_qwen3next_fattn.py")
    
    if not os.path.exists(test_script):
        print(f"Test script not found: {test_script}")
        return 1
    
    result = subprocess.run([sys.executable, test_script], cwd=test_dir)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())

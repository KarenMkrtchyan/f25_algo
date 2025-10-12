#!/usr/bin/env python3
"""
Test runner for interpretability functions.
This script runs all test cases and provides a summary.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
from tests.test_simple import run_simple_tests


def main():
    """Main test runner function."""
    print("🧪 Interpretability Functions Test Suite")
    print("=" * 50)
    
    # Run simple tests
    print("\n📋 Running Basic Function Tests...")
    success = run_simple_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        print("\n✅ Your interpretability functions are working correctly.")
        print("📝 You can now use these functions in your experiments.")
    else:
        print("⚠️  Some tests failed.")
        print("🔧 Please check the error messages above and fix any issues.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

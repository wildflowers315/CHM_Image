#!/usr/bin/env python3
"""
Test runner for CHM Image Processing tests.

This script provides different test execution modes for development and CI.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_command(cmd, timeout=300):
    """Run command with timeout and capture output."""
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(__file__))  # Run from project root
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️  Completed in {elapsed:.2f}s")
        
        if result.returncode == 0:
            print("✅ PASSED")
            if result.stdout.strip():
                print(result.stdout)
        else:
            print("❌ FAILED")
            if result.stderr.strip():
                print("STDERR:", result.stderr)
            if result.stdout.strip():
                print("STDOUT:", result.stdout)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pytest', 'numpy', 'rasterio', 'torch'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def create_test_data():
    """Create synthetic test data if needed."""
    print("🎲 Creating test data...")
    
    # Add current directory to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fixtures'))
    
    try:
        from synthetic_data import create_test_datasets
        datasets = create_test_datasets()
        print(f"✅ Created {len(datasets)} test datasets")
        return True
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return False

def run_unit_tests(verbose=False):
    """Run unit tests."""
    print("\n📋 Running unit tests...")
    
    cmd = [sys.executable, '-m', 'pytest', 'tests/unit/', '-v']
    if not verbose:
        cmd.append('-q')
    
    return run_command(cmd, timeout=120)

def run_integration_tests(verbose=False):
    """Run integration tests."""
    print("\n🔗 Running integration tests...")
    
    cmd = [sys.executable, '-m', 'pytest', 'tests/integration/', '-v']
    if not verbose:
        cmd.append('-q')
    
    return run_command(cmd, timeout=180)

def run_quick_tests(verbose=False):
    """Run quick subset of tests."""
    print("\n⚡ Running quick tests...")
    
    # Run specific quick tests
    quick_tests = [
        'tests/unit/test_spatial_utils_enhanced.py::TestEnhancedSpatialMerger::test_initialization',
        'tests/unit/test_image_patches_enhanced.py::TestTemporalDetection::test_detect_temporal_mode_temporal',
    ]
    
    cmd = [sys.executable, '-m', 'pytest'] + quick_tests
    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
    
    return run_command(cmd, timeout=60)

def run_coverage_tests():
    """Run tests with coverage reporting."""
    print("\n📊 Running tests with coverage...")
    
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/',
        '--cov=.',
        '--cov-report=term-missing',
        '--cov-report=html:tests/htmlcov',
        '-v'
    ]
    
    return run_command(cmd, timeout=300)

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='CHM Image Processing Test Runner')
    parser.add_argument(
        'mode',
        choices=['quick', 'unit', 'integration', 'all', 'coverage'],
        help='Test execution mode'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency check'
    )
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='Skip test data creation'
    )
    
    args = parser.parse_args()
    
    print("🧪 CHM Image Processing Test Runner")
    print("=" * 50)
    
    # Check dependencies
    if not args.skip_deps:
        if not check_dependencies():
            return 1
    
    # Create test data
    if not args.skip_data:
        if not create_test_data():
            return 1
    
    # Run tests based on mode
    success = True
    
    if args.mode == 'quick':
        success = run_quick_tests(args.verbose)
    
    elif args.mode == 'unit':
        success = run_unit_tests(args.verbose)
    
    elif args.mode == 'integration':
        success = run_integration_tests(args.verbose)
    
    elif args.mode == 'all':
        success = (
            run_unit_tests(args.verbose) and
            run_integration_tests(args.verbose)
        )
    
    elif args.mode == 'coverage':
        success = run_coverage_tests()
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
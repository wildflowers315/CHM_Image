#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for CHM Image Processing tests.
"""

import pytest
import os
import sys
import warnings
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Add fixtures directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fixtures'))

# Filter common warnings during testing
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning)

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory."""
    return os.path.join(os.path.dirname(__file__), 'fixtures', 'data')

@pytest.fixture(scope="session")
def ensure_test_data(test_data_dir):
    """Ensure test data exists for the session."""
    try:
        from test_config import ensure_test_data_exists
        datasets = ensure_test_data_exists()
        return datasets
    except ImportError:
        pytest.skip("test_config not available")

@pytest.fixture
def temp_dir():
    """Provide temporary directory that's cleaned up after test."""
    temp_dir = tempfile.mkdtemp(prefix='chm_test_')
    yield temp_dir
    
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def synthetic_generator():
    """Provide synthetic data generator."""
    try:
        from synthetic_data import SyntheticDataGenerator
        return SyntheticDataGenerator(seed=42)  # Fixed seed for reproducible tests
    except ImportError:
        pytest.skip("synthetic_data not available")

@pytest.fixture
def test_config():
    """Provide test configuration."""
    try:
        from test_config import TEST_CONFIG
        return TEST_CONFIG
    except ImportError:
        pytest.skip("test_config not available")

# Configure pytest options
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (skip if no GPU)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add slow marker to tests that might be slow
        if any(keyword in item.name.lower() for keyword in ['training', 'unet', '3d', 'large']):
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if 'integration' in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker to unit tests
        if 'unit' in str(item.fspath):
            item.add_marker(pytest.mark.unit)

# Custom pytest options
def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-gpu", action="store_true", default=False,
        help="run GPU tests (if GPU available)"
    )

def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip slow tests unless explicitly requested
    if "slow" in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")
    
    # Skip GPU tests unless explicitly requested and GPU available
    if "gpu" in item.keywords:
        if not item.config.getoption("--run-gpu"):
            pytest.skip("need --run-gpu option to run")
        
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available")

@pytest.fixture(autouse=True)
def memory_cleanup():
    """Automatic memory cleanup after each test."""
    yield
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear PyTorch cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
"""
Pytest configuration and shared fixtures for all test modules.
"""
import pytest
import os
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables before running tests."""
    load_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(load_env_path):
        load_dotenv(load_env_path)


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return os.path.join(os.path.dirname(__file__), 'data')

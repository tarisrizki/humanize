import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data and override settings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_data_dir = settings.DATA_DIR
        settings.DATA_DIR = tmpdir
        yield tmpdir
        settings.DATA_DIR = original_data_dir

@pytest.fixture
def client(temp_data_dir):
    """Provide a FastAPI TestClient."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def sample_user_id():
    return "test_user_123"

@pytest.fixture
def sample_text_en():
    return "This is a sample document for testing the English analyzer. However, it is quite short."

@pytest.fixture
def sample_text_id():
    return "Ini adalah dokumen contoh untuk menguji penganalisis Bahasa Indonesia. Namun, teks ini cukup singkat."

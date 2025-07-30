# tests/test_api.py
from fastapi.testclient import TestClient
import sys
import os

# Add the parent directory to Python path so we can import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app 

# The secret token from main.py
AUTH_TOKEN = "e33ecffb686ac4409ef84a4cebb13b83e58b120976b3e7b73481e7cc3daf20de"
client = TestClient(app)

def test_read_root():
    """Tests if the server is alive."""
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]

def test_unauthorized_access():
    """Tests if the API correctly blocks requests without a valid token."""
    response = client.post("/api/v1/hackrx/run", json={"documents": "url", "questions": ["q"]})
    assert response.status_code == 403  # Forbidden (FastAPI returns 403 for missing auth)

def test_authorized_dummy_run():
    """Tests an authorized call (this will evolve as the project grows)."""
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    request_data = {
        "documents": "https://example.com/policy.pdf",
        "questions": ["What is x?"]
    }
    response = client.post("/api/v1/hackrx/run", json=request_data, headers=headers)
    assert response.status_code == 200
    assert "answers" in response.json()
    
def test_unauthorized_access():
    """Tests if the API correctly blocks requests without a valid token."""
    # Test with no token (returns 403)
    response = client.post("/api/v1/hackrx/run", json={"documents": "url", "questions": ["q"]})
    assert response.status_code == 403

def test_invalid_token():
    """Tests if the API correctly blocks requests with an invalid token."""
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.post("/api/v1/hackrx/run", json={"documents": "url", "questions": ["q"]}, headers=headers)
    assert response.status_code == 401  # Unauthorized
    
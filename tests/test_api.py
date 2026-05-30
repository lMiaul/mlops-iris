# tests/test_api.py
from fastapi.testclient import TestClient
from src.api import app

# Cliente de pruebas que simula peticiones sin encender el servidor real
client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"mensaje": "API de MLOps en línea"}

def test_predict():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    
    # Verificamos que la API no falló
    assert response.status_code == 200
    # Verificamos que la respuesta tiene la llave que esperamos
    assert "prediccion" in response.json()

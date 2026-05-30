# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator # <- NUEVO

# Inicializar la API
app = FastAPI(title="Iris ML API", description="API para predecir especies de Iris")

# Activar el monitoreo y exponer la ruta /metrics <- NUEVO
Instrumentator().instrument(app).expose(app)

# Cargar el modelo (esto ocurre una sola vez al prender el servidor)
model = joblib.load('models/model.pkl')

# Definir cómo deben ser los datos de entrada
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"mensaje": "API de MLOps en línea"}

@app.post("/predict")
def predict(features: IrisFeatures):
    # Convertir JSON a DataFrame (usando model_dump para Pydantic v2)
    data = pd.DataFrame([features.model_dump()])
    
    # ¡Importante! El modelo se entrenó con nombres de columnas específicos. 
    # Debemos renombrarlas para que coincidan.
    data.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    # Hacer predicción
    prediction = model.predict(data)
    
    # Las clases son 0: setosa, 1: versicolor, 2: virginica
    clases = {0: "setosa", 1: "versicolor", 2: "virginica"}
    resultado = clases[prediction[0]]
    
    return {"prediccion": resultado}

# src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

def main():
    # 1. Configurar MLflow para que guarde localmente en un archivo SQLite
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-classification") # Nombre de nuestro proyecto

    # 2. Cargar los datos que descargamos en la Fase 1
    if not os.path.exists('data/raw/iris.csv'):
        print("Error: No se encontró data/raw/iris.csv")
        return
    
    df = pd.read_csv('data/raw/iris.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Separar en datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Iniciar un "Run" (Experimento) en MLflow
    with mlflow.start_run():
        # Definir Hiperparámetros
        n_estimators = 100
        max_depth = 5

        # -> ¡MLOps! Guardar los hiperparámetros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Entrenar Modelo
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # Evaluar Modelo
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Precisión del modelo: {accuracy:.4f}")

        # -> ¡MLOps! Guardar las métricas de éxito
        mlflow.log_metric("accuracy", accuracy)

        # -> ¡MLOps! Guardar el modelo empaquetado (el .pkl)
        mlflow.sklearn.log_model(model, "random_forest_model")

if __name__ == "__main__":
    main()

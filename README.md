# Proyecto MLOps Integral: Clasificador Iris 🌸

Este repositorio contiene un pipeline completo de **Machine Learning Operations (MLOps)** desarrollado desde cero. El objetivo de este proyecto no es el modelo en sí (un Random Forest sencillo), sino la **arquitectura e infraestructura** construida a su alrededor para llevarlo a un entorno de producción robusto, reproducible y monitoreado.

## 🏗 Arquitectura del Proyecto

El proyecto está dividido en 5 fases lógicas que representan el ciclo de vida real de un modelo de Machine Learning:

1. **Control de Versiones de Datos (DVC + Git):** Separación de código y datos. Descarga automatizada y versionado de datos pesados (`iris.csv`) usando DVC.
2. **Seguimiento de Experimentos (MLflow):** Registro automático de hiperparámetros y métricas (`accuracy`) durante el entrenamiento, guardando el modelo final (`model.pkl`) en un Registry local.
3. **Empaquetado y Contenedorización (FastAPI + Docker):** Creación de una API REST ultra rápida para exponer el modelo y empaquetado del entorno exacto usando un contenedor de Docker.
4. **Integración Continua (GitHub Actions + Pytest):** Automatización de pruebas unitarias y validación de la construcción de la imagen Docker en cada `git push` a la rama `main`.
5. **Despliegue y Monitoreo (Docker Compose + Prometheus + Grafana):** Orquestación de servicios en una red privada virtual y generación de paneles visuales para vigilar la salud y rendimiento de la API en tiempo real.

---

## 🛠 Tecnologías Utilizadas

*   **Python 3.10**
*   **Scikit-Learn / Pandas** (Modelado y manipulación de datos)
*   **Git / DVC** (Control de versiones)
*   **MLflow** (Experiment Tracking & Model Registry)
*   **FastAPI / Uvicorn** (API REST e instrumentación)
*   **Pytest / HTTPX** (Pruebas unitarias)
*   **Docker / Docker Compose** (Contenedorización y Orquestación)
*   **GitHub Actions** (CI Pipeline)
*   **Prometheus / Grafana** (Monitoreo de métricas)

---

## 🚀 Cómo ejecutar este proyecto localmente

### 1. Clonar y configurar el entorno
```bash
git clone https://github.com/tu-usuario/mlops-iris.git
cd mlops-iris
python -m venv venv
.\venv\Scripts\Activate.ps1 # En Windows
pip install -r requirements.txt
```

### 2. Entrenamiento y Tracking (Opcional)
Si deseas re-entrenar el modelo y ver la interfaz de MLflow:
```bash
python src/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
*(Puedes ver los resultados en `http://localhost:5000`)*

### 3. Levantar Infraestructura de Producción (API + Monitoreo)
Asegúrate de tener **Docker Desktop** en ejecución y lanza toda la arquitectura orquestada:
```bash
docker-compose up --build -d
```

### 4. Puertos y Servicios Disponibles
Una vez que los contenedores estén corriendo, puedes acceder a:
*   🟢 **API Interactiva (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
*   📊 **Base de Métricas (Prometheus):** [http://localhost:9090](http://localhost:9090)
*   📈 **Panel de Monitoreo (Grafana):** [http://localhost:3000](http://localhost:3000) *(Usuario: admin / Pass: admin)*

---

## 📂 Estructura de Directorios

```text
mlops-iris/
├── .github/workflows/   # Pipelines de CI/CD (GitHub Actions)
├── data/raw/            # Datos originales (trackeados por DVC)
├── models/              # Modelo empacado para producción (.pkl)
├── src/                 
│   ├── get_data.py      # Script para descargar datos
│   ├── train.py         # Script de entrenamiento y MLflow
│   └── api.py           # Aplicación FastAPI
├── tests/               # Pruebas unitarias con Pytest
├── .gitignore           # Archivos ignorados por Git
├── docker-compose.yml   # Orquestador de servicios (API, Prometheus, Grafana)
├── Dockerfile           # Instrucciones de la imagen de la API
├── prometheus.yml       # Configuración de Prometheus
└── requirements.txt     # Dependencias del proyecto
```
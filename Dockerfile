# Usar una imagen oficial de Python ligera
FROM python:3.10-slim

# Establecer directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y la carpeta del modelo
COPY src/ src/
COPY models/ models/

# Exponer el puerto donde correrá la API
EXPOSE 8000

# Comando para ejecutar la API cuando se encienda el contenedor
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

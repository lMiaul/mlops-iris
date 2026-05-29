# src/get_data.py
import os
import pandas as pd
from sklearn.datasets import load_iris

def main():
    # Crear directorio si no existe
    os.makedirs('data/raw', exist_ok=True)
    
    # Cargar datos
    print("Descargando dataset de Iris...")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Guardar a CSV
    output_path = 'data/raw/iris.csv'
    df.to_csv(output_path, index=False)
    print(f"Datos guardados en {output_path}")

if __name__ == '__main__':
    main()

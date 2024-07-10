import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np

# Cargar los datos desde el archivo CSV
input_file = 'historical_data_with_indicators.csv'
output_file = 'datos_normalizados.csv'

# Leer el archivo CSV
df = pd.read_csv(input_file)

# Verificar y manejar valores faltantes
def handle_missing_values(df):
    # Identificar columnas con valores faltantes
    columns_with_missing_values = df.columns[df.isnull().any()]

    if len(columns_with_missing_values) > 0:
        print(f"Columnas con valores faltantes: {', '.join(columns_with_missing_values)}")

        # Imputar valores faltantes usando la media
        imputer = SimpleImputer(strategy='mean')
        df[columns_with_missing_values] = imputer.fit_transform(df[columns_with_missing_values])

    return df

# Normalizar las columnas numéricas
def normalize_numeric_columns(df):
    # Seleccionar las columnas numéricas a normalizar
    columnas_numericas = ['open', 'SMA50', 'SMA100', 'EMA12', 'EMA26']

    # Inicializar el scaler
    scaler = MinMaxScaler()

    # Normalizar los datos
    df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

    return df

# Función principal para procesar y guardar los datos normalizados
def process_and_save_data(input_file, output_file):
    # Cargar los datos
    df = pd.read_csv(input_file)

    # Manejar valores faltantes
    df = handle_missing_values(df)

    # Normalizar columnas numéricas
    df = normalize_numeric_columns(df)

    # Guardar los datos normalizados en un nuevo archivo CSV
    df.to_csv(output_file, index=False)
    print(f"Datos normalizados guardados en: {output_file}")

# Ejecutar el proceso de normalización
process_and_save_data(input_file, output_file)

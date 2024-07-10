import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# Parámetros para la obtención de datos
symbol = 'BTCUSDT'
interval = '1m'
limit = 1000
base_url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
output_file = f'{symbol}_historical_data.csv'

def main():
    try:
        # Calcular la fecha de inicio 2 meses atrás desde la fecha actual
        start_time = datetime.now() - timedelta(days=90)
        
        # Verificar si el archivo existe y leer el último timestamp almacenado
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            last_timestamp = int(df['timestamp'].iloc[-1])
            start_time = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(minutes=1)
        else:
            # Crear un DataFrame vacío si el archivo no existe
            df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Obtener datos nuevos y agregar al DataFrame
        while start_time <= datetime.now():
            new_data = get_data(start_time)
            if new_data.empty:
                print("No se obtuvieron datos nuevos. Finalizando la actualización.")
                break
            
            # Limpiar nuevos datos de NaNs antes de concatenar
            new_data = new_data.dropna()
            
            df = pd.concat([df, new_data], ignore_index=True, sort=False)
            
            # Guardar en el archivo CSV
            df.to_csv(output_file, index=False)
            print(f"Se han almacenado {len(new_data)} registros nuevos. Total de registros: {len(df)}")
            
            # Establecer el nuevo punto de inicio para la siguiente iteración
            last_timestamp = int(df['timestamp'].iloc[-1])
            start_time = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(minutes=1)
            
            # Pausa de 1 segundo antes de la próxima solicitud para no sobrecargar la API
            time.sleep(1)
    
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la solicitud a la API de Binance: {e}")
    
    except Exception as e:
        print(f"Error desconocido: {e}")

def get_data(start_time):
    url = f'{base_url}&startTime={int(start_time.timestamp() * 1000)}'
    retries = 3
    for _ in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Procesar los datos y devolver un DataFrame
            processed_data = []
            for entry in data:
                timestamp = int(entry[0])
                open_price = float(entry[1])
                high_price = float(entry[2])
                low_price = float(entry[3])
                close_price = float(entry[4])
                volume = float(entry[5])
                
                processed_data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(processed_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error al hacer la solicitud: {e}. Reintentando en 10 segundos...")
            time.sleep(10)
    
    return pd.DataFrame()

if __name__ == "__main__":
    main()

import requests
import pandas as pd
import ta
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import time
import os

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parámetros
MAX_LEVERAGE = 10
STOP_LOSS_PERCENT = 0.02  # Ejemplo de stop-loss en 2%
TAKE_PROFIT_PERCENT = 0.05  # Ejemplo de take-profit en 5%
TRADING_FEE = 0.001  # Comisión de trading

# URL del Webhook de Slack
SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/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'

# Archivo para guardar los datos históricos
HISTORICAL_DATA_FILE = 'historical_data.csv'

# Función para obtener datos históricos
def get_data(symbol='BTCUSDT', interval='1m', limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    retries = 3
    for _ in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            df.dropna(inplace=True)
            return df
        except requests.exceptions.RequestException as e:
            logging.error(f"Error al obtener datos: {e}")
            time.sleep(10)
    return pd.DataFrame()

# Función para calcular indicadores técnicos
def calculate_indicators(data):
    if data.empty:
        raise ValueError("No hay suficientes datos para calcular indicadores.")
    data['SMA50'] = data['close'].rolling(window=50).mean()
    data['SMA100'] = data['close'].rolling(window=100).mean()
    data['EMA12'] = data['close'].ewm(span=12).mean()
    data['EMA26'] = data['close'].ewm(span=26).mean()
    std = data['close'].rolling(window=20).std()
    data['Upper_BB'] = data['close'].rolling(window=20).mean() + 2 * std
    data['Lower_BB'] = data['close'].rolling(window=20).mean() - 2 * std
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9).mean()
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['MFI'] = ta.volume.money_flow_index(data['high'], data['low'], data['close'], data['volume'], window=14)
    data['High_Rolling'] = data['high'].rolling(50).max()
    data['Low_Rolling'] = data['low'].rolling(50).min()
    data['Close_Shift'] = data['close'].shift(1)
    data['Pivot'] = (data['High_Rolling'] + data['Low_Rolling'] + data['Close_Shift']) / 3
    data['R1'] = 2 * data['Pivot'] - data['Low_Rolling']
    data['S1'] = 2 * data['Pivot'] - data['High_Rolling']
    data['R2'] = data['Pivot'] + (data['High_Rolling'] - data['Low_Rolling'])
    data['S2'] = data['Pivot'] - (data['High_Rolling'] - data['Low_Rolling'])
    data['ADX'] = ta.trend.adx(data['high'], data['low'], data['close'], window=14)
    data['CCI'] = ta.trend.cci(data['high'], data['low'], data['close'], window=14)
    data['ROC'] = ta.momentum.roc(data['close'], window=12)
    
    # Crear una columna de intervalo de tiempo objetivo (interval_target)
    data['interval_target'] = data['close'].pct_change().shift(-1)
    
    data.dropna(inplace=True)
    return data

# Función para crear y entrenar modelo de Machine Learning
def create_ml_model(data):
    data['price_target'] = data['close'].pct_change().shift(-1)
    data.dropna(inplace=True)
    
    historical_window = 1000
    data = data.iloc[-historical_window:]
    
    features = ['SMA50', 'SMA100', 'EMA12', 'EMA26', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'RSI', 'MFI', 'ADX', 'CCI', 'ROC']
    X = data[features]
    y = data['price_target']

    if len(X) < 10:
        raise ValueError("No hay suficientes datos para entrenar el modelo.")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    clf = RandomForestRegressor(random_state=42)
    search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=50, cv=tscv, scoring='neg_mean_squared_error', verbose=2, random_state=42, n_jobs=-1)
    search.fit(X, y)

    best_params = search.best_params_
    logging.info(f"Mejores hiperparámetros encontrados: {best_params}")

    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X, y)

    return model

# Función para predecir múltiples puntos en el futuro
def predict_multiple_future_prices(model, data, num_predictions=5):
    last_row = data.iloc[-1]
    features = ['SMA50', 'SMA100', 'EMA12', 'EMA26', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'RSI', 'MFI', 'ADX', 'CCI', 'ROC']
    X = last_row[features].values.reshape(1, -1)
    
    future_predictions = []
    for _ in range(num_predictions):
        future_prediction = model.predict(X)[0]
        future_predictions.append(future_prediction)
        
        # Actualizar X con la nueva predicción para predecir el siguiente punto en el futuro
        X = X[:, 1:]  # Descartar la primera columna (la más antigua)
        X = np.append(X, future_prediction).reshape(1, -1)
    
    return future_predictions

# Función para calcular precios de entrada y salida recomendados
def calculate_entry_exit_prices(current_price, future_predictions):
    entry_price = current_price
    exit_price = future_predictions[-1]
    
    return entry_price, exit_price

# Función para enviar recomendaciones a Slack
def send_slack_recommendation(action, entry_price, exit_price, leverage, certainty):
    color = "#36a64f" if action == "LONG" else "#ff0000"
    
    message = {
        "attachments": [
            {
                "color": color,
                "fields": [
                    {"title": "Recomendación", "value": f"{action} BTCUSDT", "short": False},
                    {"title": "Duración recomendada", "value": "1h", "short": True},
                    {"title": "Nivel de apalancamiento", "value": f"{leverage}x", "short": True},
                    {"title": "Precio de entrada", "value": f"{entry_price:.2f} USDT", "short": True},
                    {"title": "Precio estimado de salida", "value": f"{exit_price:.2f} USDT", "short": True},
                    {"title": "Certidumbre", "value": f"{certainty:.2f}%", "short": True}
                ]
            }
        ]
    }

    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=message, timeout=10)
        response.raise_for_status()
        logging.info("Notificación enviada exitosamente")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error al enviar notificación: {e}")

def main():
    intervalo_espera = 300  # Esperar 5 minutos entre iteraciones

    while True:
        try:
            if not os.path.exists(HISTORICAL_DATA_FILE):
                with open(HISTORICAL_DATA_FILE, 'w') as f:
                    f.write("timestamp,open,high,low,close,volume,close_time,quote_asset_volume,number_of_trades,taker_buy_base_asset_volume,taker_buy_quote_asset_volume,ignore\n")

            historical_data = pd.read_csv(HISTORICAL_DATA_FILE)

            new_data = get_data()
            if not new_data.empty:
                logging.info("Nuevos datos obtenidos correctamente.")
                historical_data = pd.concat([historical_data, new_data], ignore_index=True)
                historical_data.to_csv(HISTORICAL_DATA_FILE, index=False)

                historical_data = calculate_indicators(historical_data)
                logging.info("Indicadores calculados correctamente.")

                model = create_ml_model(historical_data)
                logging.info("Modelo de ML creado correctamente.")

                current_price = historical_data['close'].iloc[-1]
                future_predictions = predict_multiple_future_prices(model, historical_data)
                entry_price, exit_price = calculate_entry_exit_prices(current_price, future_predictions)
                
                certainty = 0.70  # Simulado, ajustar según el modelo real

                action = "LONG" if exit_price > current_price else "SHORT"
                send_slack_recommendation(action, entry_price, exit_price, MAX_LEVERAGE, certainty)

            else:
                logging.error("No se obtuvieron nuevos datos.")

            logging.info("Esperando el siguiente intervalo...")
            time.sleep(intervalo_espera)

        except ValueError as ve:
            logging.error(f"Error al crear o reentrenar el modelo de Machine Learning: {ve}")

        except Exception as e:
            logging.error(f"Error desconocido: {e}")

if __name__ == "__main__":
    main()

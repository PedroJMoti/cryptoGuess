import requests
import pandas as pd
import ta
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
from imblearn.over_sampling import SMOTE
import time
import os

# Parámetros de apalancamiento y gestión de riesgos
MAX_LEVERAGE = 10  # Máximo nivel de apalancamiento
STOP_LOSS_PERCENT = 0.05  # 5% de pérdida máxima permitida
TAKE_PROFIT_PERCENT = 0.07  # 7% de ganancia objetivo

# Definir comisión por trade de Binance
TRADING_FEE = 0.001  # 0.1% de comisión

# URL del Webhook de Slack
SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/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'

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Archivo para guardar los datos históricos
HISTORICAL_DATA_FILE = 'historical_data.csv'

# Función para obtener datos históricos
def get_data(symbol='BTCUSDT', interval='1m', limit=1000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    retries = 3  # Número de reintentos en caso de error de red
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
            time.sleep(10)  # Esperar 10 segundos antes de reintentar
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
    data['interval_target'] = (data['close'].shift(-1) - data['close']) / data['close']
    
    # Convertir interval_target en categorías
    data['interval_target'] = pd.cut(data['interval_target'], bins=5, labels=False)
    
    data.dropna(inplace=True)
    return data

# Función para crear y entrenar modelo de Machine Learning
def create_ml_model(data):
    data['price_target'] = (data['close'].shift(-1) > data['close']).astype(int)
    data.dropna(inplace=True)
    
    # Incluir datos históricos previos
    historical_window = 1000  # Utilizar los últimos 1000 datos históricos
    data = data.iloc[-historical_window:]
    
    features = ['SMA50', 'SMA100', 'EMA12', 'EMA26', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'RSI', 'MFI', 'ADX', 'CCI', 'ROC']
    X = data[features]
    y_price = data['price_target']
    y_interval = data['interval_target']

    if len(X) < 10:  # Asegurar que haya al menos 10 muestras para entrenar el modelo
        raise ValueError("No hay suficientes datos para entrenar el modelo.")
    
    # Sobremuestreo para manejar desbalance de clases
    smote = SMOTE(random_state=42)
    X_resampled, y_price_resampled = smote.fit_resample(X, y_price)

    # División de datos con TimeSeriesSplit para validación cruzada
    tscv = TimeSeriesSplit(n_splits=5)

    # Definir el espacio de búsqueda de hiperparámetros para RandomizedSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Inicializar el clasificador RandomForest
    clf = RandomForestClassifier(random_state=42)

    # Usar RandomizedSearchCV para encontrar los mejores hiperparámetros
    search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=50, cv=tscv, scoring='accuracy', verbose=2, random_state=42, n_jobs=-1)
    search.fit(X_resampled, y_price_resampled)

    # Obtener los mejores hiperparámetros encontrados
    best_params = search.best_params_
    logging.info(f"Mejores hiperparámetros encontrados: {best_params}")

    # Evaluar precisión con los mejores hiperparámetros
    y_price_pred = search.predict(X_resampled)
    accuracy = accuracy_score(y_price_resampled, y_price_pred)
    logging.info(f"Precisión con mejores hiperparámetros: {accuracy:.2f}")

    # Entrenar modelo para predecir interval_target
    interval_model = RandomForestClassifier(random_state=42, **best_params)
    interval_model.fit(X, y_interval)

    return search.best_estimator_, interval_model, accuracy

# Función para realizar predicción con el modelo entrenado
def make_prediction(model, interval_model, data):
    features = ['SMA50', 'SMA100', 'EMA12', 'EMA26', 'Upper_BB', 'Lower_BB', 'MACD', 'Signal', 'RSI', 'MFI', 'ADX', 'CCI', 'ROC']
    X = data[features].iloc[-1].values.reshape(1, -1)
    
    logging.info(f"Ejemplo de datos de entrada para la predicción: {X}")
    
    price_prediction = model.predict(X)[0]
    price_certainty = max(model.predict_proba(X)[0])
    
    interval_prediction = interval_model.predict(X)[0]
    
    y_actual = data['price_target'].iloc[-1]
    y_pred = model.predict(data[features].iloc[:-1])
    precision = precision_score(data['price_target'].iloc[:-1], y_pred)

    logging.info(f"Predicción: {price_prediction}, Certidumbre: {price_certainty:.2f}, Precisión: {precision:.2f}, Intervalo: {interval_prediction}")

    return price_prediction, price_certainty, precision, interval_prediction

# Función para enviar notificación a Slack
def notify_slack(action, certainty, precision, interval_prediction, leverage):
    color = "#36a64f" if action == "BUY" else "#ff0000"
    
    interval_mapping = {0: '5m', 1: '15m', 2: '30m', 3: '1h', 4: '2h'}
    interval_text = interval_mapping.get(interval_prediction, 'Desconocido')

    trading_fee = TRADING_FEE * leverage
    precision_percentage = precision * 100
    certainty_percentage = certainty * 100
    
    message = {
        "attachments": [
            {
                "color": color,
                "fields": [
                    {"title": "Recomendación", "value": f"{action} BTCUSDT", "short": False},
                    {"title": "Duración recomendada", "value": interval_text, "short": True},
                    {"title": "Nivel de apalancamiento", "value": f"{leverage}x", "short": True},
                    {"title": "Certidumbre", "value": f"{certainty_percentage:.2f}%", "short": True},
                    {"title": "Precisión ML", "value": f"{precision_percentage:.2f}%", "short": True}
                ]
            }
        ]
    }

    logging.info(f"Enviando notificación a webhook: {SLACK_WEBHOOK_URL}")
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=message, timeout=10)
        response.raise_for_status()
        logging.info("Notificación enviada exitosamente")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error al enviar notificación: {e}")
    except Exception as e:
        logging.error(f"Error desconocido al enviar notificación: {e}")

# Función principal
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

                model, interval_model, accuracy = create_ml_model(historical_data)
                logging.info("Modelo de ML creado correctamente.")

                last_row = historical_data.iloc[-1]
                current_price = last_row['close']
                prediction, certainty, precision, interval_prediction = make_prediction(model, interval_model, historical_data)
                logging.info(f"Predicción realizada: {prediction}, Certidumbre: {certainty:.2f}, Precisión: {precision:.2f}, Intervalo: {interval_prediction}")

                action = "BUY" if prediction == 1 else "SELL"
                notify_slack(action, certainty, precision, interval_prediction, MAX_LEVERAGE)

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

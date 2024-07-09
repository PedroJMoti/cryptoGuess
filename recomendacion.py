import requests
import pandas as pd
import ta
import time
from datetime import datetime, timedelta

# Parámetros de apalancamiento y gestión de riesgos
MAX_LEVERAGE = 10  # Máximo nivel de apalancamiento
STOP_LOSS_PERCENT = 0.02  # 2% de pérdida máxima permitida
TAKE_PROFIT_PERCENT = 0.05  # 5% de ganancia objetivo

# URLs de los Webhooks de Slack
SLACK_WEBHOOK_URL = 'https://hooks.slack.com/services/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'
SLACK_BUY_SELL_WEBHOOK_URL = 'https://hooks.slack.com/services/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'

# Función para obtener datos históricos
def get_data(symbol='BTCUSDT', interval='1m', limit=50000):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    # Convertir columnas numéricas a tipo float
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # Eliminar filas con datos faltantes
    df.dropna(inplace=True)
    return df

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

# Función para crear recomendaciones de trading y enviar notificaciones a Slack
def create_and_notify_recommendations(data):
    last_row = data.iloc[-1]
    sma50_above_sma100 = last_row['SMA50'] > last_row['SMA100']
    sma50_cross_sma100 = data.iloc[-2]['SMA50'] < data.iloc[-2]['SMA100'] and sma50_above_sma100
    macd_above_signal = last_row['MACD'] > last_row['Signal']
    macd_cross_signal = data.iloc[-2]['MACD'] < data.iloc[-2]['Signal'] and macd_above_signal
    rsi = last_row['RSI']
    mfi = last_row['MFI']
    price = last_row['close']
    upper_band = last_row['Upper_BB']
    lower_band = last_row['Lower_BB']
    
    recommendation = 'HOLD'
    certainty = 0
    hold_time = 0
    leverage = 1  # Apalancamiento base

    # Evaluación de recomendaciones
    if sma50_cross_sma100:
        recommendation = 'BUY'
        certainty += 0.2
        hold_time += 1  # Mantener por 1 minuto como base

    if macd_cross_signal:
        recommendation = 'BUY'
        certainty += 0.3
        hold_time += 1  # Mantener por 1 minuto como base

    if rsi < 30:
        recommendation = 'BUY'
        certainty += 0.25
        hold_time += 1  # Mantener por 1 minuto

    if mfi < 20:
        recommendation = 'BUY'
        certainty += 0.25
        hold_time += 1  # Mantener por 1 minuto

    if price > upper_band:
        recommendation = 'SELL'
        certainty += 0.2
        hold_time += 1  # Mantener por 1 minuto como base

    if price < lower_band:
        recommendation = 'BUY'
        certainty += 0.3
        hold_time += 1  # Mantener por 1 minuto como base

    # Normalizar el grado de certeza y tiempo de retención
    certainty = min(certainty, 1.0)  # Certeza máxima del 100%

    # Calcular el apalancamiento basado en la certeza
    leverage = 1 + int(certainty * (MAX_LEVERAGE - 1))

    # Enviar notificación a Slack siempre que haya una recomendación nueva
    send_slack_notification(recommendation, hold_time, leverage, certainty, price)

def send_slack_notification(recommendation, hold_time, leverage, certainty, price):
    if recommendation == 'BUY':
        color = "#008000"  # Verde para BUY
    elif recommendation == 'SELL':
        color = "#FF0000"  # Rojo para SELL
    else:
        color = ""  # Negro u otro color para cualquier otra recomendación no reconocida

    print(color)

    
    # Verificar si la certeza es alta
    if certainty >= 0.8:
        title = "**🚨 Notificación Prioritaria 🚨**"
        color = "#FFA500"  # Color naranja para notificaciones prioritarias
    else:
        title = "**Notificación de trading**"
    
    recommendation_text = f"*{recommendation}*"
    message = f"{title}\n\n*Recomendación de trading:* {recommendation_text}\n*Tiempo de retención estimado:* {hold_time} minutos\n*Apalancamiento sugerido:* {leverage}x\n*Grado de certeza:* {certainty:.2f}\n*Precio actual de Bitcoin:* ${price:.2f}"
    payload = {
        "attachments": [
            {
                "color": color,
                "text": message
            }
        ]
    }

    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code == 200:
            print("Notificación enviada exitosamente a Slack.")
        else:
            print(f"Error al enviar notificación a Slack: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error de conexión al enviar notificación a Slack: {str(e)}")

    # Enviar notificación a un canal diferente para 'BUY' y 'SELL'
    if recommendation in ['BUY', 'SELL']:
        try:
            response = requests.post(SLACK_BUY_SELL_WEBHOOK_URL, json=payload)
            if response.status_code == 200:
                print(f"Notificación de {recommendation} enviada exitosamente al canal de BUY/SELL en Slack.")
            else:
                print(f"Error al enviar notificación de {recommendation} al canal de BUY/SELL en Slack: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Error de conexión al enviar notificación de {recommendation} al canal de BUY/SELL en Slack: {str(e)}")

# Función para verificar que el script sigue funcionando
def check_script_status():
    print("El script está en ejecución y funcionando correctamente.")

# Función principal del script
def main():
    try:
        print("Iniciando script de recomendaciones de trading...")
        while True:
            # Verificar que el script sigue funcionando
            check_script_status()

            # Obtener los datos históricos
            data = get_data()

            # Calcular los indicadores técnicos
            calculate_indicators(data)

            # Crear recomendaciones y enviar notificación a Slack
            create_and_notify_recommendations(data)

            # Esperar 10 segundos antes de la próxima iteración
            time.sleep(10)

    except KeyboardInterrupt:
        print("Proceso interrumpido por el usuario.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    main()

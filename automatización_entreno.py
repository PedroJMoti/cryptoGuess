import os
import time
import subprocess
import numpy as np
import csv

# Función para ejecutar un comando en la terminal y esperar a que termine
def ejecutar_comando(comando):
    result = subprocess.run(comando, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error al ejecutar el comando: {comando}")
        print(result.stderr.decode('utf-8'))
    else:
        print(result.stdout.decode('utf-8'))

# Función para calcular métricas después de cada ciclo de entrenamiento
def calcular_metricas(loss, predictions, y_test):
    mae = np.mean(np.abs(predictions - y_test))
    return mae

# Directorio donde se encuentran los scripts
directorio_scripts = '/home/pj/crypto-project/cryptoguess2.0/'


# Definir intervalo de tiempo (en segundos) para la actualización
intervalo_actualizacion = 600  # 10 minutos

# Listas para almacenar métricas
loss_history = []
mae_history = []

while True:
    try:
        # Mostrar la hora de inicio del ciclo de actualización y reentrenamiento
        print(f"Iniciando actualización y reentrenamiento a las {time.strftime('%Y-%m-%d %H:%M:%S')}")


        # Ejecutar script para calcular indicadores
        obtener_datos_script = os.path.join(directorio_scripts, 'obtener_datos_lastyear.py')
        ejecutar_comando(f'python {obtener_datos_script}')

        # Ejecutar script para calcular indicadores
        calcular_indicadores_script = os.path.join(directorio_scripts, 'calcular_indicadores.py')
        ejecutar_comando(f'python {calcular_indicadores_script}')

        # Esperar un momento para asegurar que el archivo se ha creado completamente
        time.sleep(10)

        # Ejecutar script para normalizar datos
        normalizar_datos_script = os.path.join(directorio_scripts, 'normalizar_datos.py')
        ejecutar_comando(f'python {normalizar_datos_script}')

        # Esperar un momento para asegurar que el archivo se ha creado completamente
        time.sleep(10)

        # Ejecutar script para entrenar el modelo
        entrenamiento_script = os.path.join(directorio_scripts, 'entrenamiento.py')
        ejecutar_comando(f'python {entrenamiento_script}')

        # Leer las métricas del entrenamiento desde el archivo o el output
        # Asumiendo que se guarda la pérdida (loss) en un archivo o se imprime en la salida estándar
        loss = 227.8542022705078  # reemplazar con el valor real obtenido
        loss_history.append(loss)

        # Simular predicciones para calcular el MAE
        predictions = np.random.rand(100) * 200  # reemplazar con las predicciones reales obtenidas
        y_test = np.random.rand(100) * 200  # reemplazar con los valores reales de prueba obtenidos

        # Calcular el error absoluto medio (MAE)
        mae = calcular_metricas(loss, predictions, y_test)
        mae_history.append(mae)

        # Mostrar métricas de entrenamiento
        print(f'Loss en datos de prueba: {loss}')
        print(f'Error absoluto medio: {mae}')

        # Guardar métricas en un archivo CSV
        with open('metricas_entrenamiento.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), loss, mae])

         # Esperar un momento para asegurar que el archivo se ha creado completamente
        time.sleep(10)

         # Ejecutar script para predecir
        prediccion_script = os.path.join(directorio_scripts, 'cryptoguess2.0.py')
        ejecutar_comando(f'python {prediccion_script}')
        print('Recomendación guardada')

         # Esperar un momento para asegurar que el archivo de recomendacion se ha creado completamente
        time.sleep(10)

         # Ejecutar script para predecir
        notificacion_script = os.path.join(directorio_scripts, 'notificacion.py')
        ejecutar_comando(f'python {notificacion_script}')
        print('Notificacion enviada a slack')
        

        # Mostrar la hora de finalización del ciclo de actualización y reentrenamiento
        print(f"Finalizando actualización y reentrenamiento a las {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Esperar el intervalo de tiempo para la próxima actualización
        print(f"Esperando {intervalo_actualizacion} segundos para la próxima actualización...")
        time.sleep(intervalo_actualizacion)

    except KeyboardInterrupt:
        print("Proceso de actualización y reentrenamiento detenido manualmente.")
        break
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        time.sleep(60)  # Esperar 1 minuto antes de intentar de nuevo en caso de error

# Al finalizar el script, podrías guardar las métricas históricas (loss_history y mae_history) en archivos CSV o bases de datos para análisis y seguimiento a largo plazo.

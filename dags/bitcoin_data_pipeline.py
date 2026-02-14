"""
DAG para descargar datos históricos y precio actual de Bitcoin
Autor: Tu Canal de YouTube
Descripción: Pipeline completo para alimentar nuestro modelo de predicción de BTC
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants.constants import HISTORICAL_PATH, CURRENT_PATH, REPORTS_PATH
from src.download_data.current import descargar_precio_actual
from src.download_data.historical import descargar_datos_historicos
from src.download_data.report import generar_reporte_html
from src.download_data.validate_current import validar_precio_actual
from src.download_data.validate_historical import validar_datos_historicos



# Definir argumentos por defecto del DAG
default_args = {
    'owner': 'youtube_tutorial',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,  # Reintentar 2 veces si falla
    'retry_delay': timedelta(minutes=5),
}

# Crear el DAG
with DAG(
    'bitcoin_data_pipeline',
    default_args=default_args,
    description='Pipeline completo para descargar y validar datos de Bitcoin',
    schedule_interval='0 9 * * *',  # Corre todos los días a las 9 AM
    catchup=False,  # No ejecutar fechas pasadas
    tags=['bitcoin', 'ml', 'data-engineering'],
) as dag:

    # Task 1: Crear directorios
    crear_directorios = BashOperator(
        task_id='crear_directorios',
        bash_command=f'mkdir -p {HISTORICAL_PATH} {CURRENT_PATH} {REPORTS_PATH}',
    )

    # Task 2: Descargar datos históricos
    descargar_historicos = PythonOperator(
        task_id='descargar_historicos',
        python_callable=descargar_datos_historicos,
        provide_context=True,
    )

    # Task 3: Validar datos históricos
    validar_historicos = PythonOperator(
        task_id='validar_historicos',
        python_callable=validar_datos_historicos,
        provide_context=True,
    )

    # Task 4: Descargar precio actual
    descargar_actual = PythonOperator(
        task_id='descargar_precio_actual',
        python_callable=descargar_precio_actual,
        provide_context=True,
    )

    # Task 5: Validar precio actual
    validar_actual = PythonOperator(
        task_id='validar_precio_actual',
        python_callable=validar_precio_actual,
        provide_context=True,
    )

    # Task 6: Generar reporte
    generar_reporte = PythonOperator(
        task_id='generar_reporte',
        python_callable=generar_reporte_html,
        provide_context=True,
    )


    # Definir dependencias (flujo del DAG)
    crear_directorios >> descargar_historicos >> validar_historicos >> generar_reporte
    crear_directorios >> descargar_actual >> validar_actual >> generar_reporte

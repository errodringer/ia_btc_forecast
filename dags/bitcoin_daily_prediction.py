"""
DAG de Predicción Diaria Automatizada de Bitcoin
Autor: Errodringer
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants.constants import PREDICTIONS_PATH
from src.model_predictions.create_ft_to_predict import crear_features_para_prediccion
from src.model_predictions.download_current_data import descargar_datos_recientes
from src.model_predictions. report import generar_reporte_diario
from src.model_predictions.send_notification import enviar_notificacion
from src.model_predictions.today_predictions import hacer_prediccion_hoy


# Definir argumentos por defecto
default_args = {
    'owner': 'Errodringer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Crear el DAG
with DAG(
    'bitcoin_daily_prediction',
    default_args=default_args,
    description='Pipeline de predicción diaria automatizada de Bitcoin',
    schedule_interval=None,
    catchup=False,
    tags=['bitcoin', 'ml', 'prediction', 'production'],
) as dag:
    
    crear_directorios = BashOperator(
        task_id='crear_directorios',
        bash_command=f'mkdir -p {PREDICTIONS_PATH}',
    )
    
    descargar_datos = PythonOperator(
        task_id='descargar_datos_recientes',
        python_callable=descargar_datos_recientes,
        provide_context=True,
    )
    
    crear_features = PythonOperator(
        task_id='crear_features',
        python_callable=crear_features_para_prediccion,
        provide_context=True,
    )
    
    hacer_prediccion = PythonOperator(
        task_id='hacer_prediccion',
        python_callable=hacer_prediccion_hoy,
        provide_context=True,
    )
    
    notificar = PythonOperator(
        task_id='enviar_notificacion',
        python_callable=enviar_notificacion,
        provide_context=True,
    )
    
    generar_reporte = PythonOperator(
        task_id='generar_reporte',
        python_callable=generar_reporte_diario,
        provide_context=True,
    )
    
    crear_directorios >> descargar_datos >> crear_features >> hacer_prediccion >> [notificar, generar_reporte]

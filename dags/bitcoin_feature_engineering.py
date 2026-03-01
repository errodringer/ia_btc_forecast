"""
DAG de Feature Engineering para Bitcoin
Autor: Errodringer
"""
import sys

from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants.constants import PROCESSED_PATH, FEATURES_PATH
from src.ft_engineering.load_data import cargar_datos_historicos
from src.ft_engineering.clean_data import limpiar_datos
from src.ft_engineering.create_tech_ft import crear_features_tecnicas
from src.ft_engineering.create_temp_ft import crear_features_temporales
from src.ft_engineering.create_target import crear_target_variable
from src.ft_engineering.prepare_dataset import preparar_dataset_final
from src.ft_engineering.report import generar_reporte_features


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
    'bitcoin_feature_engineering',
    default_args=default_args,
    description='Pipeline de feature engineering para Bitcoin ML',
    schedule_interval=None,  # Manual trigger (corre después del pipeline de datos)
    catchup=False,
    tags=['bitcoin', 'ml', 'feature-engineering'],
) as dag:
    
    crear_directorios = BashOperator(
        task_id='crear_directorios',
        bash_command=f'mkdir -p {PROCESSED_PATH} {FEATURES_PATH}',
    )

    cargar_datos = PythonOperator(
        task_id='cargar_datos',
        python_callable=cargar_datos_historicos,
        provide_context=True,
    )

    limpiar = PythonOperator(
        task_id='limpiar_datos',
        python_callable=limpiar_datos,
        provide_context=True,
    )

    crear_tecnicas = PythonOperator(
        task_id='crear_features_tecnicas',
        python_callable=crear_features_tecnicas,
        provide_context=True,
    )

    crear_temporales = PythonOperator(
        task_id='crear_features_temporales',
        python_callable=crear_features_temporales,
        provide_context=True,
    )

    crear_target = PythonOperator(
        task_id='crear_target_variable',
        python_callable=crear_target_variable,
        provide_context=True,
    )

    preparar_dataset = PythonOperator(
        task_id='preparar_dataset_final',
        python_callable=preparar_dataset_final,
        provide_context=True,
    )

    generar_reporte = PythonOperator(
        task_id='generar_reporte',
        python_callable=generar_reporte_features,
        provide_context=True,
    )

    trigger_model_training = TriggerDagRunOperator(
        task_id='trigger_model_training',
        trigger_dag_id='bitcoin_model_training',
        wait_for_completion=False,  # True si quieres esperar a que termine el segundo DAG
        reset_dag_run=True,         # Opcional: reinicia si ya existe un run
    )

    crear_directorios >> cargar_datos >> limpiar >> crear_tecnicas >> crear_temporales >> crear_target >> preparar_dataset >> generar_reporte >> trigger_model_training

"""
DAG de Entrenamiento de Modelos ML para Bitcoin
Autor: Errodringer
"""
from pathlib import Path
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants.constants import PLOTS_PATH, MODELS_PATH
from src.model_training.load_dataset import cargar_datasets
from src.model_training.ft_normalize import normalizar_features
from src.model_training.train_logistic_regression import entrenar_logistic_regression
from src.model_training.train_random_forest import entrenar_random_forest
from src.model_training.train_gradient_boosting import entrenar_gradient_boosting
from src.model_training.evaluate_models import evaluar_modelos_en_test
from src.model_training.generate_plots import generar_graficos
from src.model_training.report import generar_reporte_final


# Definir argumentos por defecto
default_args = {
    'owner': 'Errodringer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}

# Crear el DAG
with DAG(
    'bitcoin_model_training',
    default_args=default_args,
    description='Pipeline de entrenamiento de modelos ML para Bitcoin',
    schedule_interval=None,  # Manual trigger
    catchup=False,
    tags=['bitcoin', 'ml', 'training'],
) as dag:

    crear_directorios = BashOperator(
        task_id='crear_directorios',
        bash_command=f'mkdir -p {PLOTS_PATH} {MODELS_PATH}',
    )

    cargar_data = PythonOperator(
        task_id='cargar_datasets',
        python_callable=cargar_datasets,
        provide_context=True,
    )

    normalizar = PythonOperator(
        task_id='normalizar_features',
        python_callable=normalizar_features,
        provide_context=True,
    )

    entrenar_lr = PythonOperator(
        task_id='entrenar_logistic_regression',
        python_callable=entrenar_logistic_regression,
        provide_context=True,
    )

    entrenar_rf = PythonOperator(
        task_id='entrenar_random_forest',
        python_callable=entrenar_random_forest,
        provide_context=True,
    )

    entrenar_gb = PythonOperator(
        task_id='entrenar_gradient_boosting',
        python_callable=entrenar_gradient_boosting,
        provide_context=True,
    )

    evaluar = PythonOperator(
        task_id='evaluar_modelos',
        python_callable=evaluar_modelos_en_test,
        provide_context=True,
    )

    graficos = PythonOperator(
        task_id='generar_graficos',
        python_callable=generar_graficos,
        provide_context=True,
    )

    reporte = PythonOperator(
        task_id='generar_reporte',
        python_callable=generar_reporte_final,
        provide_context=True,
    )

    cargar_data >> normalizar
    normalizar >> [entrenar_lr, entrenar_rf, entrenar_gb]
    [entrenar_lr, entrenar_rf, entrenar_gb] >> evaluar >> graficos >> reporte

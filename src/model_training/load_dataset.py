import logging
import pandas as pd

from src.constants.constants import FEATURES_PATH



def cargar_datasets(**context):
    """
    Cargar los datasets de train y test generados en la etapa de feature engineering
    """
    logging.info("📂 Cargando datasets de entrenamiento y prueba...")

    # Verificar que existen los archivos
    archivos_requeridos = [
        'X_train.parquet', 'X_test.parquet',
        'y_train.parquet', 'y_test.parquet',
        'prices_train.parquet', 'prices_test.parquet'
    ]

    for archivo in archivos_requeridos:
        filepath = FEATURES_PATH / archivo
        if not filepath.exists():
            raise FileNotFoundError(f"❌ Archivo no encontrado: {filepath}")

    # Cargar datasets
    X_train = pd.read_parquet(FEATURES_PATH / "X_train.parquet")
    X_test = pd.read_parquet(FEATURES_PATH / "X_test.parquet")
    y_train = pd.read_parquet(FEATURES_PATH / "y_train.parquet")['target']
    y_test = pd.read_parquet(FEATURES_PATH / "y_test.parquet")['target']

    logging.info(f"✅ Datasets cargados exitosamente")
    logging.info(f"📊 X_train shape: {X_train.shape}")
    logging.info(f"📊 X_test shape: {X_test.shape}")
    logging.info(f"📊 y_train shape: {y_train.shape}")
    logging.info(f"📊 y_test shape: {y_test.shape}")

    # Verificar balance de clases
    train_balance = y_train.value_counts()
    test_balance = y_test.value_counts()

    logging.info(
        f"📊 Balance train - Sube: {train_balance[1]} ({train_balance[1]/len(y_train)*100:.1f}%), "
        f"Baja: {train_balance[0]} ({train_balance[0]/len(y_train)*100:.1f}%)"
    )
    logging.info(
        f"📊 Balance test - Sube: {test_balance[1]} ({test_balance[1]/len(y_test)*100:.1f}%), "
        f"Baja: {test_balance[0]} ({test_balance[0]/len(y_test)*100:.1f}%)"
    )

    # Guardar metadata
    context['task_instance'].xcom_push(key='train_size', value=len(X_train))
    context['task_instance'].xcom_push(key='test_size', value=len(X_test))
    context['task_instance'].xcom_push(key='num_features', value=X_train.shape[1])

    return True

if __name__ == "__main__":
    # Para pruebas locales
    cargar_datasets()

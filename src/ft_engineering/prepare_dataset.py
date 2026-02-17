import logging
import pandas as pd

from src.constants.constants import FEATURES_PATH

def preparar_dataset_final(**context):
    """
    Preparar el dataset final: eliminar NaNs, seleccionar features, split train/test
    """
    logging.info("🎁 Preparando dataset final...")

    with_target_file = context['task_instance'].xcom_pull(
        task_ids='crear_target_variable',
        key='with_target_file'
    )

    df = pd.read_parquet(with_target_file)

    registros_antes = len(df)
    logging.info(f"📊 Registros antes de limpieza: {registros_antes}")

    # Eliminar filas con NaN (principalmente al inicio por las ventanas móviles)
    df = df.dropna()

    registros_despues = len(df)
    logging.info(f"📊 Registros después de limpieza: {registros_despues}")
    logging.info(f"🗑️ Registros eliminados: {registros_antes - registros_despues}")

    # Seleccionar features para el modelo
    # Excluir: date, precio original, adj close, y las variables target (excepto una)
    columnas_excluir = ['date', 'open', 'high', 'low', 'close', 'volume', 'target_direction',
                        'adj close', 'target_next_close', 'target_pct_change']

    feature_columns = [col for col in df.columns if col not in columnas_excluir]

    logging.info(f"📋 Features seleccionadas: {len(feature_columns)}")

    # Crear dataset con features y target
    X = df[feature_columns].copy()
    y = df['target_direction'].copy()  # Usamos dirección como target

    # También guardamos los precios para referencia
    prices = df[['date', 'close', 'target_next_close']].copy()

    # Split temporal: 80% train, 20% test
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    prices_train = prices.iloc[:split_index]
    prices_test = prices.iloc[split_index:]

    logging.info(f"📊 Train set: {len(X_train)} registros")
    logging.info(f"📊 Test set: {len(X_test)} registros")
    logging.info(f"📅 Train periodo: {prices_train['date'].min()} a {prices_train['date'].max()}")
    logging.info(f"📅 Test periodo: {prices_test['date'].min()} a {prices_test['date'].max()}")

    # Guardar datasets
    X_train.to_parquet(FEATURES_PATH / "X_train.parquet", index=False)
    X_test.to_parquet(FEATURES_PATH / "X_test.parquet", index=False)
    y_train.to_frame('target').to_parquet(FEATURES_PATH / "y_train.parquet", index=False)
    y_test.to_frame('target').to_parquet(FEATURES_PATH / "y_test.parquet", index=False)
    prices_train.to_parquet(FEATURES_PATH / "prices_train.parquet", index=False)
    prices_test.to_parquet(FEATURES_PATH / "prices_test.parquet", index=False)

    # Guardar lista de features
    with open(FEATURES_PATH / "feature_names.txt", 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    logging.info(f"✅ Datasets guardados en: {FEATURES_PATH}")
    logging.info("📁 Archivos creados:")
    logging.info("   - X_train.parquet")
    logging.info("   - X_test.parquet")
    logging.info("   - y_train.parquet")
    logging.info("   - y_test.parquet")
    logging.info("   - prices_train.parquet")
    logging.info("   - prices_test.parquet")
    logging.info("   - feature_names.txt")

    # Metadata para siguiente pipeline
    context['task_instance'].xcom_push(key='num_features', value=len(feature_columns))
    context['task_instance'].xcom_push(key='train_size', value=len(X_train))
    context['task_instance'].xcom_push(key='test_size', value=len(X_test))

    return True


if __name__ == "__main__":
    # Prueba local
    preparar_dataset_final()

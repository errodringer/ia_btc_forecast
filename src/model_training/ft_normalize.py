import logging
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler

from src.constants.constants import FEATURES_PATH, MODELS_PATH


def normalizar_features(**context):
    """
    Normalizar features usando StandardScaler
    Muy importante: fit solo en train, transform en ambos
    """
    logging.info("📏 Normalizando features con StandardScaler...")

    # Cargar datos
    X_train = pd.read_parquet(FEATURES_PATH / "X_train.parquet")
    X_test = pd.read_parquet(FEATURES_PATH / "X_test.parquet")

    logging.info(f"📊 Rango antes de normalizar (ejemplo 'close'):")
    if 'close' in X_train.columns:
        logging.info(f"   Train: [{X_train['close'].min():.2f}, {X_train['close'].max():.2f}]")

    # Inicializar scaler
    scaler = StandardScaler()

    # CRÍTICO: Fit solo en train (para evitar data leakage)
    scaler.fit(X_train)

    # Transform ambos datasets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertir de vuelta a DataFrames
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    logging.info(f"✅ Features normalizadas")
    logging.info(f"📊 Media después de normalizar: {X_train_scaled.mean().mean():.4f} (debería ser ~0)")
    logging.info(f"📊 Std después de normalizar: {X_train_scaled.std().mean():.4f} (debería ser ~1)")

    # Guardar datasets normalizados
    X_train_scaled.to_parquet(FEATURES_PATH / "X_train_scaled.parquet", index=False)
    X_test_scaled.to_parquet(FEATURES_PATH / "X_test_scaled.parquet", index=False)

    # Guardar scaler para uso futuro
    with open(MODELS_PATH / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    logging.info(f"💾 Scaler guardado en: {MODELS_PATH / 'scaler.pkl'}")

    return True


if __name__ == "__main__":
    # Para pruebas locales
    normalizar_features()

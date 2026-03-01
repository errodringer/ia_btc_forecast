import logging
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

from src.constants.constants import FEATURES_PATH, MODELS_PATH


def entrenar_logistic_regression(**context):
    """
    Entrenar modelo de Logistic Regression (baseline simple)
    """
    logging.info("🤖 Entrenando Logistic Regression...")

    # Cargar datos normalizados
    X_train = pd.read_parquet(FEATURES_PATH / "X_train_scaled.parquet")
    y_train = pd.read_parquet(FEATURES_PATH / "y_train.parquet")['target']

    # Entrenar modelo
    model = LogisticRegression(
        max_iter=100,
        random_state=42,
        n_jobs=1,
        fit_intercept=True,
        class_weight='balanced'  # Para manejar desbalance de clases
    )

    logging.info("🔄 Entrenando modelo...")
    model.fit(X_train, y_train)

    # Predicciones en train
    y_train_pred = model.predict(X_train)

    # Métricas en train
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    logging.info("✅ Modelo entrenado")
    logging.info("📊 Métricas en TRAIN:")
    logging.info(f"   Accuracy: {train_accuracy:.4f}")
    logging.info(f"   Precision: {train_precision:.4f}")
    logging.info(f"   Recall: {train_recall:.4f}")
    logging.info(f"   F1-Score: {train_f1:.4f}")

    # Guardar modelo
    model_path = MODELS_PATH / "logistic_regression.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    logging.info(f"💾 Modelo guardado en: {model_path}")

    # Guardar métricas
    metrics = {
        'model': 'Logistic Regression',
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1
    }

    context['task_instance'].xcom_push(key='lr_metrics', value=metrics)

    return metrics

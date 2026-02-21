import logging
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.constants.constants import FEATURES_PATH, MODELS_PATH


def entrenar_random_forest(**context):
    """
    Entrenar Random Forest Classifier
    """
    logging.info("🌳 Entrenando Random Forest...")

    # Cargar datos normalizados
    X_train = pd.read_parquet(FEATURES_PATH / "X_train_scaled.parquet")
    y_train = pd.read_parquet(FEATURES_PATH / "y_train.parquet")['target']

    # Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        min_samples_split=0.05,
        min_samples_leaf=0.01,
        random_state=42,
        class_weight='balanced',
        warm_start=False,
        max_features=0.5,
        oob_score=True,
        n_jobs=1
    )

    logging.info("🔄 Entrenando modelo (puede tardar un poco)...")
    model.fit(X_train, y_train)

    # Predicciones en train
    y_train_pred = model.predict(X_train)

    # Métricas en train
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    logging.info(f"✅ Modelo entrenado")
    logging.info(f"📊 Métricas en TRAIN:")
    logging.info(f"   Accuracy: {train_accuracy:.4f}")
    logging.info(f"   Precision: {train_precision:.4f}")
    logging.info(f"   Recall: {train_recall:.4f}")
    logging.info(f"   F1-Score: {train_f1:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logging.info(f"🔝 Top 10 features más importantes:")
    for idx, row in feature_importance.head(10).iterrows():
        logging.info(f"   {row['feature']}: {row['importance']:.4f}")

    # Guardar feature importance
    feature_importance.to_csv(MODELS_PATH / "rf_feature_importance.csv", index=False)

    # Guardar modelo
    model_path = MODELS_PATH / "random_forest.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    logging.info(f"💾 Modelo guardado en: {model_path}")

    # Guardar métricas
    metrics = {
        'model': 'Random Forest',
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1
    }

    context['task_instance'].xcom_push(key='rf_metrics', value=metrics)

    return metrics

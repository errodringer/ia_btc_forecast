import logging
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.constants.constants import FEATURES_PATH, MODELS_PATH


def entrenar_gradient_boosting(**context):
    """
    Entrenar Gradient Boosting Classifier
    """
    logging.info("⚡ Entrenando Gradient Boosting...")

    # Cargar datos normalizados
    X_train = pd.read_parquet(FEATURES_PATH / "X_train_scaled.parquet")
    y_train = pd.read_parquet(FEATURES_PATH / "y_train.parquet")['target']

    # Entrenar modelo
    model = GradientBoostingClassifier(
        n_estimators=5,
        learning_rate=0.8,
        max_depth=6,
        min_samples_split=15,
        min_samples_leaf=10,
        random_state=42,
        warm_start=True,
        verbose=0,
        subsample=0.6,
        validation_fraction=0.2,
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

    logging.info("✅ Modelo entrenado")
    logging.info("📊 Métricas en TRAIN:")
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
    feature_importance.to_csv(MODELS_PATH / "gb_feature_importance.csv", index=False)

    # Guardar modelo
    model_path = MODELS_PATH / "gradient_boosting.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    logging.info(f"💾 Modelo guardado en: {model_path}")

    # Guardar métricas
    metrics = {
        'model': 'Gradient Boosting',
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_f1': train_f1
    }

    context['task_instance'].xcom_push(key='gb_metrics', value=metrics)

    return metrics

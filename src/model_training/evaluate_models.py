import json
import logging
import pickle

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.constants.constants import FEATURES_PATH, MODELS_PATH


def evaluar_modelos_en_test(**context):
    """
    Evaluar todos los modelos en el conjunto de test
    """
    logging.info("📊 Evaluando todos los modelos en TEST set...")

    # Cargar datos de test
    X_test = pd.read_parquet(FEATURES_PATH / "X_test_scaled.parquet")
    y_test = pd.read_parquet(FEATURES_PATH / "y_test.parquet")['target']

    # Modelos a evaluar
    modelos = {
        'Logistic Regression': MODELS_PATH / "logistic_regression.pkl",
        'Random Forest': MODELS_PATH / "random_forest.pkl",
        'Gradient Boosting': MODELS_PATH / "gradient_boosting.pkl"
    }

    resultados = []

    for nombre, path in modelos.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"🤖 Evaluando: {nombre}")
        logging.info(f"{'='*60}")

        # Cargar modelo
        with open(path, 'rb') as f:
            model = pickle.load(f)

        # Predicciones
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        logging.info(f"📊 Métricas en TEST:")
        logging.info(f"   Accuracy: {accuracy:.4f}")
        logging.info(f"   Precision: {precision:.4f}")
        logging.info(f"   Recall: {recall:.4f}")
        logging.info(f"   F1-Score: {f1:.4f}")
        logging.info(f"   ROC-AUC: {roc_auc:.4f}")

        logging.info(f"\n📊 Confusion Matrix:")
        logging.info(f"   TN: {cm[0,0]}  FP: {cm[0,1]}")
        logging.info(f"   FN: {cm[1,0]}  TP: {cm[1,1]}")

        # Interpretación de negocio
        correct_predictions = cm[0,0] + cm[1,1]
        total_predictions = len(y_test)

        # Predicciones correctas de subida
        correct_ups = cm[1,1]
        total_ups = y_test.sum()

        # Predicciones correctas de bajada
        correct_downs = cm[0,0]
        total_downs = len(y_test) - y_test.sum()

        logging.info("\n💰 Interpretación de Trading:")
        logging.info(
            f"   Predicciones correctas: {correct_predictions}/{total_predictions} "
            f"({correct_predictions/total_predictions*100:.1f}%)"
        )
        logging.info(
            f"   Subidas correctas: {correct_ups}/{total_ups} "
            f"({correct_ups/total_ups*100:.1f}%)"
        )
        logging.info(
            f"   Bajadas correctas: {correct_downs}/{total_downs} "
            f"({correct_downs/total_downs*100:.1f}%)"
        )

        # Guardar resultados
        resultado = {
            'model': nombre,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'correct_predictions': int(correct_predictions),
            'total_predictions': int(total_predictions),
            'correct_ups': int(correct_ups),
            'total_ups': int(total_ups),
            'correct_downs': int(correct_downs),
            'total_downs': int(total_downs)
        }

        resultados.append(resultado)

    # Guardar todos los resultados
    with open(MODELS_PATH / "test_results.json", 'w') as f:
        json.dump(resultados, f, indent=2)

    logging.info(f"\n✅ Evaluación completa guardada en: {MODELS_PATH / 'test_results.json'}")

    # Determinar mejor modelo
    mejor_modelo = max(resultados, key=lambda x: x['accuracy'])
    logging.info(f"\n🏆 MEJOR MODELO: {mejor_modelo['model']}")
    logging.info(f"   Accuracy: {mejor_modelo['accuracy']:.4f}")

    # Mapeo de nombres a rutas de archivos
    modelo_paths = {
        'Logistic Regression': MODELS_PATH / "logistic_regression.pkl",
        'Random Forest': MODELS_PATH / "random_forest.pkl",
        'Gradient Boosting': MODELS_PATH / "gradient_boosting.pkl"
    }
    
    # Cargar el mejor modelo desde su archivo
    mejor_modelo_nombre = mejor_modelo['model']
    mejor_modelo_path = modelo_paths[mejor_modelo_nombre]
    
    logging.info(f"📂 Cargando mejor modelo desde: {mejor_modelo_path}")
    
    with open(mejor_modelo_path, 'rb') as f:
        best_model = pickle.load(f)
    
    # Guardar como best_model.pkl
    best_model_save_path = MODELS_PATH / "best_model.pkl"
    with open(best_model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    logging.info(f"✅ Mejor modelo guardado como: {best_model_save_path}")

    context['task_instance'].xcom_push(key='mejor_modelo', value=mejor_modelo['model'])
    context['task_instance'].xcom_push(key='resultados', value=resultados)

    return resultados

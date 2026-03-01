import json
import logging

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Para generar gráficos sin display
import matplotlib.pyplot as plt
import seaborn as sns

from src.constants.constants import MODELS_PATH, PLOTS_PATH


def generar_graficos(**context):
    """
    Generar gráficos de comparación de modelos
    """
    logging.info("📊 Generando gráficos de evaluación...")

    # Cargar resultados
    with open(MODELS_PATH / "test_results.json", 'r') as f:
        resultados = json.load(f)

    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 1. Comparación de métricas
    logging.info("📊 Gráfico 1: Comparación de métricas...")

    fig, ax = plt.subplots(figsize=(12, 6))

    modelos = [r['model'] for r in resultados]
    metricas = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    x = np.arange(len(modelos))
    width = 0.15

    for i, metrica in enumerate(metricas):
        valores = [r[metrica] for r in resultados]
        ax.bar(x + i*width, valores, width, label=metrica.upper())

    ax.set_xlabel('Modelos', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(modelos, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "metricas_comparacion.png", dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"✅ Guardado: {PLOTS_PATH / 'metricas_comparacion.png'}")

    # 2. Confusion matrices
    logging.info("📊 Gráfico 2: Confusion matrices...")

    _, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, resultado in enumerate(resultados):
        cm = np.array(resultado['confusion_matrix'])

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Baja', 'Sube'],
            yticklabels=['Baja', 'Sube'],
            ax=axes[idx],
            cbar=False
        )

        axes[idx].set_title(resultado['model'], fontweight='bold')
        axes[idx].set_ylabel('Real', fontweight='bold')
        axes[idx].set_xlabel('Predicción', fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"✅ Guardado: {PLOTS_PATH / 'confusion_matrices.png'}")

    # 3. Feature importance (Random Forest)
    logging.info("📊 Gráfico 3: Feature importance...")

    if (MODELS_PATH / "rf_feature_importance.csv").exists():
        fi = pd.read_csv(MODELS_PATH / "rf_feature_importance.csv")

        fig, ax = plt.subplots(figsize=(10, 8))

        top_features = fi.head(20)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importancia', fontweight='bold')
        ax.set_title('Top 20 Features Más Importantes (Random Forest)', fontweight='bold', fontsize=14)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(PLOTS_PATH / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"✅ Guardado: {PLOTS_PATH / 'feature_importance.png'}")

    logging.info("✅ Todos los gráficos generados")

    return True

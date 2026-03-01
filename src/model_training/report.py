import json
import logging
from datetime import datetime

from src.constants.constants import REPORTS_PATH, MODELS_PATH


def generar_reporte_final(**context):
    """
    Generar reporte HTML final con todos los resultados
    """
    logging.info("📄 Generando reporte final...")

    # Cargar resultados
    with open(MODELS_PATH / "test_results.json", 'r') as f:
        resultados = json.load(f)

    mejor_modelo = context['task_instance'].xcom_pull(
        task_ids='evaluar_modelos',
        key='mejor_modelo'
    )

    # Crear HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bitcoin ML - Resultados del Entrenamiento</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }}
            .container {{
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            h1 {{
                color: #667eea;
                text-align: center;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #764ba2;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
                margin-top: 30px;
            }}
            .success {{
                background: #10b981;
                color: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                margin: 20px 0;
                font-size: 18px;
            }}
            .winner {{
                background: gold;
                color: #333;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                margin: 20px 0;
                font-size: 20px;
                font-weight: bold;
                border: 3px solid #ffd700;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background: #667eea;
                color: white;
            }}
            tr:hover {{
                background: #f5f5f5;
            }}
            .best-model {{
                background: #fffacd !important;
                font-weight: bold;
            }}
            .metric-good {{
                color: #10b981;
                font-weight: bold;
            }}
            .metric-medium {{
                color: #f59e0b;
                font-weight: bold;
            }}
            .metric-bad {{
                color: #ef4444;
                font-weight: bold;
            }}
            .chart-container {{
                margin: 30px 0;
                text-align: center;
            }}
            .chart-container img {{
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Resultados del Entrenamiento - Bitcoin ML</h1>

            <div class="success">
                ✅ Entrenamiento completado - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            
            <div class="winner">
                🏆 MEJOR MODELO: {mejor_modelo}
            </div>
            
            <h2>📊 Comparación de Modelos en TEST</h2>
            
            <table>
                <tr>
                    <th>Modelo</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>ROC-AUC</th>
                </tr>
    """

    for resultado in resultados:
        is_best = resultado['model'] == mejor_modelo
        row_class = 'best-model' if is_best else ''

        # Colorear métricas según valor
        def color_metric(value):
            if value >= 0.6:
                return 'metric-good'
            elif value >= 0.5:
                return 'metric-medium'
            else:
                return 'metric-bad'

        html_content += f"""
                <tr class="{row_class}">
                    <td>{"🏆 " if is_best else ""}{resultado['model']}</td>
                    <td class="{color_metric(resultado['accuracy'])}">{resultado['accuracy']:.4f}</td>
                    <td class="{color_metric(resultado['precision'])}">{resultado['precision']:.4f}</td>
                    <td class="{color_metric(resultado['recall'])}">{resultado['recall']:.4f}</td>
                    <td class="{color_metric(resultado['f1'])}">{resultado['f1']:.4f}</td>
                    <td class="{color_metric(resultado['roc_auc'])}">{resultado['roc_auc']:.4f}</td>
                </tr>
        """

    html_content += """
            </table>
            
            <h2>💰 Interpretación de Trading</h2>
            
            <table>
                <tr>
                    <th>Modelo</th>
                    <th>Predicciones Correctas</th>
                    <th>Subidas Correctas</th>
                    <th>Bajadas Correctas</th>
                </tr>
    """

    for resultado in resultados:
        is_best = resultado['model'] == mejor_modelo
        row_class = 'best-model' if is_best else ''

        correct_pct = resultado['correct_predictions'] / resultado['total_predictions'] * 100
        ups_pct = resultado['correct_ups'] / resultado['total_ups'] * 100
        downs_pct = resultado['correct_downs'] / resultado['total_downs'] * 100

        html_content += f"""
                <tr class="{row_class}">
                    <td>{"🏆 " if is_best else ""}{resultado['model']}</td>
                    <td>{resultado['correct_predictions']}/{resultado['total_predictions']} ({correct_pct:.1f}%)</td>
                    <td>{resultado['correct_ups']}/{resultado['total_ups']} ({ups_pct:.1f}%)</td>
                    <td>{resultado['correct_downs']}/{resultado['total_downs']} ({downs_pct:.1f}%)</td>
                </tr>
        """

    html_content += f"""
            </table>
            
            <h2>📊 Gráficos de Evaluación</h2>
            
            <div class="chart-container">
                <h3>Comparación de Métricas</h3>
                <img src="plots/metricas_comparacion.png" alt="Comparación de Métricas">
            </div>
            
            <div class="chart-container">
                <h3>Matrices de Confusión</h3>
                <img src="plots/confusion_matrices.png" alt="Confusion Matrices">
            </div>
            
            <div class="chart-container">
                <h3>Features Más Importantes</h3>
                <img src="plots/feature_importance.png" alt="Feature Importance">
            </div>
            
            <h2>💡 Interpretación de Resultados</h2>
            
            <div style="background: #e0e7ff; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3>¿Qué significan estas métricas?</h3>
                <ul>
                    <li><strong>Accuracy:</strong> % de predicciones correctas totales</li>
                    <li><strong>Precision:</strong> De lo que predijimos "sube", ¿cuánto realmente subió?</li>
                    <li><strong>Recall:</strong> De lo que realmente subió, ¿cuánto detectamos?</li>
                    <li><strong>F1-Score:</strong> Balance entre Precision y Recall</li>
                    <li><strong>ROC-AUC:</strong> Capacidad del modelo de distinguir clases (0.5 = azar, 1.0 = perfecto)</li>
                </ul>
            </div>
            
            <div style="background: #fef3c7; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3>🎯 ¿Es bueno este resultado?</h3>
                <p>Para trading de Bitcoin:</p>
                <ul>
                    <li>✅ Accuracy > 55% es mejor que azar (50%)</li>
                    <li>✅ F1-Score > 0.55 indica que el modelo tiene señal útil</li>
                    <li>⚠️ ROC-AUC entre 0.55-0.65 es "modesto pero útil"</li>
                    <li>🎯 ROC-AUC > 0.70 sería excelente (pero raro en cripto)</li>
                </ul>
                <p><strong>Recuerda:</strong> Bitcoin es extremadamente volátil y difícil de predecir. 
                Incluso un modelo con 55-60% de accuracy puede ser rentable con buena gestión de riesgo.</p>
            </div>
            
            <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 2px solid #eee; color: #666;">
                <p>Modelos guardados en: /opt/airflow/data/models/</p>
                <p>Pipeline ejecutado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Guardar reporte
    report_path = REPORTS_PATH / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logging.info(f"✅ Reporte generado: {report_path}")

    return str(report_path)

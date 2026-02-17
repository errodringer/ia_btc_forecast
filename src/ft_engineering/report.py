import logging
from datetime import datetime
import pandas as pd

from src.constants.constants import BASE_PATH, PROCESSED_PATH


def generar_reporte_features(**context):
    """
    Generar reporte HTML con estadísticas de las features creadas
    """
    logging.info("📊 Generando reporte de features...")

    # Obtener metadata
    num_technical = context['task_instance'].xcom_pull(
        task_ids='crear_features_tecnicas',
        key='num_technical_features'
    )
    num_temporal = context['task_instance'].xcom_pull(
        task_ids='crear_features_temporales',
        key='num_temporal_features'
    )
    num_features = context['task_instance'].xcom_pull(
        task_ids='preparar_dataset_final',
        key='num_features'
    )
    train_size = context['task_instance'].xcom_pull(
        task_ids='preparar_dataset_final',
        key='train_size'
    )
    test_size = context['task_instance'].xcom_pull(
        task_ids='preparar_dataset_final',
        key='test_size'
    )

    # Cargar datos para estadísticas
    df = pd.read_parquet(PROCESSED_PATH / "btc_with_target.parquet")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Engineering Report - Bitcoin</title>
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
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .stat-card h3 {{
                margin: 0 0 10px 0;
                font-size: 14px;
                opacity: 0.9;
            }}
            .stat-card .value {{
                font-size: 32px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .feature-list {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 10px;
                margin: 20px 0;
            }}
            .feature-item {{
                background: #f0f4f8;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
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
            .info-box {{
                background: #e0e7ff;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Feature Engineering - Bitcoin ML</h1>

            <div class="success">
                ✅ Pipeline de Feature Engineering completado - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            
            <h2>📊 Resumen del Dataset</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>🎯 FEATURES TOTALES</h3>
                    <div class="value">{num_features}</div>
                </div>
                
                <div class="stat-card">
                    <h3>📈 FEATURES TÉCNICAS</h3>
                    <div class="value">{num_technical}</div>
                </div>
                
                <div class="stat-card">
                    <h3>📅 FEATURES TEMPORALES</h3>
                    <div class="value">{num_temporal}</div>
                </div>
                
                <div class="stat-card">
                    <h3>📚 TRAIN SET</h3>
                    <div class="value">{train_size:,}</div>
                </div>
                
                <div class="stat-card">
                    <h3>🧪 TEST SET</h3>
                    <div class="value">{test_size:,}</div>
                </div>
                
                <div class="stat-card">
                    <h3>📊 SPLIT RATIO</h3>
                    <div class="value">{train_size/(train_size+test_size)*100:.0f}% / {test_size/(train_size+test_size)*100:.0f}%</div>
                </div>
            </div>
            
            <h2>📈 Features Técnicas Creadas</h2>
            <div class="info-box">
                <strong>Indicadores técnicos</strong> basados en análisis de precios históricos
            </div>
            
            <table>
                <tr>
                    <th>Categoría</th>
                    <th>Features</th>
                    <th>Descripción</th>
                </tr>
                <tr>
                    <td>📊 Medias Móviles</td>
                    <td>SMA-7, SMA-14, SMA-21, SMA-50, SMA-200, EMA-12, EMA-26</td>
                    <td>Tendencias a corto, medio y largo plazo</td>
                </tr>
                <tr>
                    <td>💪 Momentum</td>
                    <td>RSI-14, MACD, MACD Signal, MACD Histogram</td>
                    <td>Fuerza y dirección del movimiento</td>
                </tr>
                <tr>
                    <td>📏 Volatilidad</td>
                    <td>Bollinger Bands, ATR, Volatility (7/14/30 días)</td>
                    <td>Medidas de riesgo y variabilidad</td>
                </tr>
                <tr>
                    <td>📊 Volumen</td>
                    <td>Volume SMA, Volume Ratio</td>
                    <td>Actividad del mercado</td>
                </tr>
            </table>
            
            <h2>📅 Features Temporales Creadas</h2>
            <div class="info-box">
                <strong>Patrones temporales</strong> para capturar estacionalidad y ciclos
            </div>
            
            <div class="feature-list">
                <div class="feature-item">📅 Día de la semana</div>
                <div class="feature-item">📆 Día del mes</div>
                <div class="feature-item">📊 Semana del año</div>
                <div class="feature-item">🗓️ Mes</div>
                <div class="feature-item">📈 Trimestre</div>
                <div class="feature-item">📅 Año</div>
                <div class="feature-item">🔄 Día sin/cos (cíclico)</div>
                <div class="feature-item">🔄 Mes sin/cos (cíclico)</div>
                <div class="feature-item">🌅 Es fin de semana</div>
                <div class="feature-item">📅 Inicio de mes</div>
                <div class="feature-item">📅 Fin de mes</div>
                <div class="feature-item">📊 Inicio de trimestre</div>
                <div class="feature-item">📊 Fin de trimestre</div>
            </div>
            
            <h2>🎯 Variable Objetivo</h2>
            <div class="info-box">
                Predecimos si el precio <strong>sube o baja</strong> al día siguiente (clasificación binaria)
            </div>
            
            <table>
                <tr>
                    <th>Métrica</th>
                    <th>Valor</th>
                </tr>
                <tr>
                    <td>Días que sube</td>
                    <td>{df['target_direction'].sum()} ({df['target_direction'].mean()*100:.1f}%)</td>
                </tr>
                <tr>
                    <td>Días que baja</td>
                    <td>{len(df) - df['target_direction'].sum()} ({(1-df['target_direction'].mean())*100:.1f}%)</td>
                </tr>
                <tr>
                    <td>Cambio % promedio</td>
                    <td>{df['target_pct_change'].mean():.2f}%</td>
                </tr>
            </table>
            
            <h2>📁 Archivos Generados</h2>
            <div class="info-box">
                <strong>Listos para entrenar el modelo de ML</strong>
            </div>
            
            <table>
                <tr>
                    <th>Archivo</th>
                    <th>Descripción</th>
                    <th>Shape</th>
                </tr>
                <tr>
                    <td>X_train.parquet</td>
                    <td>Features de entrenamiento</td>
                    <td>({train_size}, {num_features})</td>
                </tr>
                <tr>
                    <td>X_test.parquet</td>
                    <td>Features de prueba</td>
                    <td>({test_size}, {num_features})</td>
                </tr>
                <tr>
                    <td>y_train.parquet</td>
                    <td>Target de entrenamiento</td>
                    <td>({train_size}, 1)</td>
                </tr>
                <tr>
                    <td>y_test.parquet</td>
                    <td>Target de prueba</td>
                    <td>({test_size}, 1)</td>
                </tr>
                <tr>
                    <td>feature_names.txt</td>
                    <td>Lista de features</td>
                    <td>{num_features} features</td>
                </tr>
            </table>
            
            <div class="success" style="margin-top: 30px;">
                🎉 ¡Dataset listo para entrenar el modelo de Machine Learning!
            </div>
            
            <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 2px solid #eee; color: #666;">
                <p>Pipeline ejecutado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Guardar reporte
    report_path = BASE_PATH / "reports" / f"feature_engineering_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logging.info(f"✅ Reporte generado: {report_path}")

    return str(report_path)


if __name__ == "__main__":
    # Prueba local
    generar_reporte_features()

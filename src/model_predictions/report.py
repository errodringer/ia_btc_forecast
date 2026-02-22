import logging
import json
from datetime import datetime

from src.constants.constants import PREDICTIONS_PATH


def generar_reporte_diario(**context):
    """
    Genera un reporte HTML con la predicción del día
    """
    logging.info("📄 Generando reporte HTML diario...")
    
    prediccion = context['task_instance'].xcom_pull(
        task_ids='hacer_prediccion',
        key='prediccion'
    )
    
    # Leer historial
    historial_file = PREDICTIONS_PATH / "historial_predicciones.jsonl"
    historial = []
    if historial_file.exists():
        with open(historial_file, 'r') as f:
            historial = [json.loads(line) for line in f.readlines()[-10:]]  # Últimas 10
    
    # Crear HTML
    emoji = "📈" if prediccion['prediccion'] == 1 else "📉"
    color = "#10b981" if prediccion['prediccion'] == 1 else "#ef4444"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predicción Bitcoin - {datetime.now().strftime('%Y-%m-%d')}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 900px;
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
            }}
            .prediccion-card {{
                background: {color};
                color: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin: 30px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .prediccion-card h2 {{
                margin: 0;
                font-size: 48px;
            }}
            .prediccion-card p {{
                margin: 10px 0;
                font-size: 24px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-box {{
                background: #f0f4f8;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }}
            .stat-box h3 {{
                margin: 0 0 10px 0;
                color: #667eea;
            }}
            .stat-box .value {{
                font-size: 28px;
                font-weight: bold;
                color: #333;
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
            .disclaimer {{
                background: #fef3c7;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #f59e0b;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Predicción Diaria de Bitcoin</h1>
            <p style="text-align: center; color: #666;">
                {datetime.now().strftime('%A, %d de %B de %Y - %H:%M:%S')}
            </p>
            
            <div class="prediccion-card">
                <h2>{emoji}</h2>
                <h2>{prediccion['direccion']}</h2>
                <p>Confianza: {prediccion['confianza']:.1%}</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>💰 Precio Actual</h3>
                    <div class="value">${prediccion['precio_hoy']:,.2f}</div>
                </div>
                
                <div class="stat-box">
                    <h3>📊 Prob. Subida</h3>
                    <div class="value">{prediccion['probabilidad_sube']:.1%}</div>
                </div>
                
                <div class="stat-box">
                    <h3>📉 Prob. Bajada</h3>
                    <div class="value">{prediccion['probabilidad_baja']:.1%}</div>
                </div>
                
                <div class="stat-box">
                    <h3>🤖 Modelo</h3>
                    <div class="value" style="font-size: 18px;">{prediccion['modelo_usado']}</div>
                </div>
            </div>
            
            <h2>🔍 Indicadores Técnicos Actuales</h2>
            <table>
                <tr>
                    <th>Indicador</th>
                    <th>Valor</th>
                </tr>
                <tr>
                    <td>RSI-14</td>
                    <td>{prediccion['features_importantes'].get('rsi_14', 'N/A'):.2f}</td>
                </tr>
                <tr>
                    <td>Volatilidad 7 días</td>
                    <td>{prediccion['features_importantes'].get('volatility_7', 0)*100:.2f}%</td>
                </tr>
                <tr>
                    <td>SMA-7</td>
                    <td>${prediccion['features_importantes'].get('sma_7', 0):,.2f}</td>
                </tr>
            </table>
    """
    
    # Agregar historial si existe
    if historial:
        html_content += """
            <h2>📊 Últimas 10 Predicciones</h2>
            <table>
                <tr>
                    <th>Fecha</th>
                    <th>Precio</th>
                    <th>Predicción</th>
                    <th>Confianza</th>
                </tr>
        """
        
        for pred in reversed(historial):
            emoji_hist = "📈" if pred['prediccion'] == 1 else "📉"
            fecha_hist = datetime.fromisoformat(pred['fecha_hoy']).strftime('%Y-%m-%d')
            html_content += f"""
                <tr>
                    <td>{fecha_hist}</td>
                    <td>${pred['precio_hoy']:,.2f}</td>
                    <td>{emoji_hist} {pred['direccion']}</td>
                    <td>{pred['confianza']:.1%}</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
    
    html_content += f"""
            <div class="disclaimer">
                <strong>⚠️ Disclaimer:</strong> Esta predicción es generada por un modelo de Machine Learning
                entrenado con datos históricos. No constituye consejo financiero. Las criptomonedas son
                altamente volátiles y riesgosas. Siempre haz tu propia investigación (DYOR) antes de invertir.
            </div>
            
            <p style="text-align: center; color: #666; margin-top: 30px;">
                Generado automáticamente por Airflow ML Pipeline
            </p>
        </div>
    </body>
    </html>
    """
    
    # Guardar reporte
    report_file = PREDICTIONS_PATH / f"reporte_diario_{datetime.now().strftime('%Y%m%d')}.html"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"✅ Reporte generado: {report_file}")
    
    return str(report_file)

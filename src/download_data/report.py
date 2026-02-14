import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yfinance as yf
import pandas as pd
import logging

from datetime import datetime

from src.constants.constants import CURRENT_PATH, REPORTS_PATH


def generar_reporte_html(**context):
    """
    Genera un reporte HTML con visualizaciones de los datos descargados
    """
    logging.info("📊 Generando reporte HTML...")
    
    # Obtener datos de XCom
    historical_file = context['task_instance'].xcom_pull(
        task_ids='descargar_historicos',
        key='historical_file'
    )
    # historical_file = "/Users/errodringer/Proyectos/ia_btc_forecast/data/historical/btc_historical_20260214.parquet"
    current_price = context['task_instance'].xcom_pull(
        task_ids='descargar_precio_actual',
        key='current_price'
    )
    # current_price = 50000.00
    historical_records = context['task_instance'].xcom_pull(
        task_ids='descargar_historicos',
        key='historical_records'
    )
    # historical_records = 732
    
    # Leer datos históricos
    df = pd.read_parquet(historical_file)
    
    # Calcular estadísticas
    precio_max = df['high'].max()
    precio_min = df['low'].min()
    precio_promedio = df['close'].mean()
    volatilidad = df['close'].std()
    
    # Generar HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte Bitcoin - {datetime.now().strftime('%Y-%m-%d')}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
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
            .stats {{
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
                font-size: 28px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .stat-card .change {{
                font-size: 14px;
                opacity: 0.8;
            }}
            .emoji {{
                font-size: 24px;
                margin-right: 10px;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 2px solid #eee;
                color: #666;
            }}
            .success {{
                background: #10b981;
                color: white;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Pipeline de Bitcoin - Reporte de Datos</h1>
            
            <div class="success">
                ✅ Pipeline ejecutado exitosamente - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <h3><span class="emoji">💰</span>PRECIO ACTUAL</h3>
                    <div class="value">${current_price:,.2f}</div>
                    <div class="change">En tiempo real</div>
                </div>
                
                <div class="stat-card">
                    <h3><span class="emoji">📊</span>REGISTROS HISTÓRICOS</h3>
                    <div class="value">{historical_records:,}</div>
                    <div class="change">Últimos 2 años</div>
                </div>
                
                <div class="stat-card">
                    <h3><span class="emoji">📈</span>PRECIO MÁXIMO</h3>
                    <div class="value">${precio_max:,.2f}</div>
                    <div class="change">All-time (2 años)</div>
                </div>
                
                <div class="stat-card">
                    <h3><span class="emoji">📉</span>PRECIO MÍNIMO</h3>
                    <div class="value">${precio_min:,.2f}</div>
                    <div class="change">All-time (2 años)</div>
                </div>
                
                <div class="stat-card">
                    <h3><span class="emoji">💵</span>PRECIO PROMEDIO</h3>
                    <div class="value">${precio_promedio:,.2f}</div>
                    <div class="change">Media histórica</div>
                </div>
                
                <div class="stat-card">
                    <h3><span class="emoji">📊</span>VOLATILIDAD</h3>
                    <div class="value">${volatilidad:,.2f}</div>
                    <div class="change">Desviación estándar</div>
                </div>
            </div>
            
            <div class="footer">
                <p><strong>Archivos generados:</strong></p>
                <p>📁 Históricos: {historical_file}</p>
                <p>📁 Actual: {CURRENT_PATH}</p>
                <p><strong>¿Precio actual vs promedio histórico?</strong></p>
                <p style="font-size: 20px; margin: 10px 0;">
                    {'+' if current_price > precio_promedio else '-'}
                    ${abs(current_price - precio_promedio):,.2f} 
                    ({abs((current_price - precio_promedio) / precio_promedio * 100):.1f}%)
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Guardar reporte
    report_filename = f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = REPORTS_PATH / report_filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"✅ Reporte generado: {report_path}")
    logging.info(f"📊 Precio actual vs promedio: {'+' if current_price > precio_promedio else '-'}${abs(current_price - precio_promedio):,.2f}")
    
    return str(report_path)


if __name__ == "__main__":
    generar_reporte_html()
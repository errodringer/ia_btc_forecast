"""
DAG de Predicción Diaria Automatizada de Bitcoin
Autor: Errodringer
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from pathlib import Path
import logging
import pickle
import json

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.preprocessing import StandardScaler

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants.constants import HISTORICAL_PATH, MODELS_PATH, PREDICTIONS_PATH, PROCESSED_PATH, FEATURES_PATH

# Crear directorios
PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)


def descargar_datos_recientes(**context):
    """
    Descarga los últimos 200 días de Bitcoin para crear features
    """
    logging.info("📥 Descargando datos recientes de Bitcoin...")
    
    try:
        # Descargar últimos 200 días (necesitamos historia para las features)
        ticker = "BTC-USD"
        btc_data = yf.download(
            ticker,
            period="200d",
            interval="1d",
            progress=False
        )
        
        if btc_data.empty:
            raise ValueError("❌ No se descargaron datos")
        
        # Preparar datos
        btc_data.reset_index(inplace=True)
        # btc_data.columns = btc_data.columns.str.lower()
        # Aquí está el cambio clave - si tienes multiindex en columnas:
        if isinstance(btc_data.columns, pd.MultiIndex):
            btc_data.columns = [
                ' '.join(col).strip().split(' ', maxsplit=1)[0] 
                for col in btc_data.columns.values
            ]
        btc_data.columns = btc_data.columns.str.lower()
        
        logging.info(f"✅ Descargados {len(btc_data)} días")
        logging.info(f"📅 Desde {btc_data['date'].min()} hasta {btc_data['date'].max()}")
        logging.info(f"💰 Precio actual: ${btc_data['close'].iloc[-1]:,.2f}")
        
        # Guardar
        output_file = PREDICTIONS_PATH / "btc_recent.parquet"
        btc_data.to_parquet(output_file, index=False)
        
        context['task_instance'].xcom_push(key='recent_file', value=str(output_file))
        context['task_instance'].xcom_push(key='precio_actual', value=float(btc_data['close'].iloc[-1]))
        
        return str(output_file)
        
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        raise


def crear_features_para_prediccion(**context):
    """
    Crea las mismas features que usamos en entrenamiento
    """
    logging.info("🔧 Creando features para predicción...")
    
    recent_file = context['task_instance'].xcom_pull(
        task_ids='descargar_datos_recientes',
        key='recent_file'
    )
    
    df = pd.read_parquet(recent_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    logging.info("📊 Creando features técnicas...")
    
    # ===== MEDIAS MÓVILES =====
    for periodo in [7, 14, 21, 50, 200]:
        df[f'sma_{periodo}'] = df['close'].rolling(window=periodo).mean()
    
    for periodo in [12, 26]:
        df[f'ema_{periodo}'] = df['close'].ewm(span=periodo, adjust=False).mean()
    
    # ===== RSI =====
    def calcular_rsi(data, periodo=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi_14'] = calcular_rsi(df['close'], 14)
    
    # ===== MACD =====
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # ===== BANDAS DE BOLLINGER =====
    periodo_bb = 20
    df['bb_middle'] = df['close'].rolling(window=periodo_bb).mean()
    bb_std = df['close'].rolling(window=periodo_bb).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    
    # ===== VOLATILIDAD =====
    for periodo in [7, 14, 30]:
        df[f'volatility_{periodo}'] = df['close'].pct_change().rolling(window=periodo).std()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    
    # ===== VOLUMEN =====
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    logging.info("� Creando features de precio histórico...")
    
    # ===== PRECIOS HISTÓRICOS =====
    # Precio de cierre de días anteriores
    for dias_atras in [1, 2, 3]:
        df[f'close_lag_{dias_atras}'] = df['close'].shift(dias_atras)
    
    # Diferencia absoluta respecto a días anteriores
    for dias_atras in [1, 2, 3]:
        df[f'price_diff_{dias_atras}'] = df['close'] - df[f'close_lag_{dias_atras}']
    
    # Cambio porcentual respecto a días anteriores
    for dias_atras in [1, 2, 3]:
        df[f'pct_change_{dias_atras}'] = (
            (df['close'] - df[f'close_lag_{dias_atras}']) / 
            df[f'close_lag_{dias_atras}'] * 100
        )
    
    # Mínimo y máximo de los últimos 3, 7, 14 días
    for periodo in [3, 7, 14]:
        df[f'min_close_{periodo}d'] = df['close'].rolling(window=periodo).min()
        df[f'max_close_{periodo}d'] = df['close'].rolling(window=periodo).max()
        df[f'dist_to_min_{periodo}d'] = df['close'] - df[f'min_close_{periodo}d']
        df[f'dist_to_max_{periodo}d'] = df['close'] - df[f'max_close_{periodo}d']
    
    # Retorno de días anteriores
    for dias_atras in [1, 2, 3]:
        df[f'return_{dias_atras}d'] = (
            (df['close'] - df[f'close_lag_{dias_atras}']) / 
            df[f'close_lag_{dias_atras}']
        )
    
    logging.info("�📅 Creando features temporales...")
    
    # ===== TEMPORALES =====
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    # Cíclicas
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Binarias
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    
    logging.info(f"✅ Features creadas: {df.shape[1]} columnas")
    
    # Guardar
    output_file = PREDICTIONS_PATH / "btc_with_features.parquet"
    df.to_parquet(output_file, index=False)
    
    context['task_instance'].xcom_push(key='features_file', value=str(output_file))
    
    return str(output_file)


def hacer_prediccion_hoy(**context):
    """
    Hace la predicción para mañana usando el modelo entrenado
    """
    logging.info("🔮 Haciendo predicción para MAÑANA...")
    
    # Cargar datos con features
    features_file = context['task_instance'].xcom_pull(
        task_ids='crear_features',
        key='features_file'
    )
    
    # features_file = PROCESSED_PATH / "btc_with_all_features.parquet"
    df = pd.read_parquet(features_file)
    
    # Tomar el último día (hoy)
    ultimo_dia = df.iloc[-1].copy()
    fecha_hoy = ultimo_dia['date']
    precio_hoy = ultimo_dia['close']
    
    logging.info(f"📅 Fecha HOY: {fecha_hoy.strftime('%Y-%m-%d')}")
    logging.info(f"💰 Precio HOY: ${precio_hoy:,.2f}")
    
    # Cargar lista de features del entrenamiento
    with open(FEATURES_PATH / "feature_names.txt", 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Seleccionar solo las features que usó el modelo
    features_para_modelo = ultimo_dia[feature_names].values.reshape(1, -1)
    
    # Cargar scaler
    with open(MODELS_PATH / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Normalizar
    features_normalizadas = scaler.transform(features_para_modelo)
    
    # Cargar modelo (Random Forest por defecto)
    modelo_path = MODELS_PATH / "random_forest.pkl"
    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)
    
    logging.info(f"✅ Modelo cargado: {modelo_path.name}")
    
    # HACER PREDICCIÓN
    prediccion = modelo.predict(features_normalizadas)[0]
    probabilidades = modelo.predict_proba(features_normalizadas)[0]
    
    prob_baja = probabilidades[0]
    prob_sube = probabilidades[1]
    
    # Resultado
    direccion = "SUBE 📈" if prediccion == 1 else "BAJA 📉"
    confianza = prob_sube if prediccion == 1 else prob_baja
    
    logging.info(f"\n{'='*60}")
    logging.info(f"🔮 PREDICCIÓN PARA MAÑANA:")
    logging.info(f"{'='*60}")
    logging.info(f"   Dirección: {direccion}")
    logging.info(f"   Confianza: {confianza:.1%}")
    logging.info(f"   Probabilidad SUBE: {prob_sube:.1%}")
    logging.info(f"   Probabilidad BAJA: {prob_baja:.1%}")
    logging.info(f"{'='*60}\n")
    
    # Crear resultado
    resultado = {
        'fecha_prediccion': datetime.now().isoformat(),
        'fecha_hoy': fecha_hoy.isoformat(),
        'fecha_manana': (fecha_hoy + timedelta(days=1)).isoformat(),
        'precio_hoy': float(precio_hoy),
        'prediccion': int(prediccion),
        'direccion': direccion,
        'probabilidad_sube': float(prob_sube),
        'probabilidad_baja': float(prob_baja),
        'confianza': float(confianza),
        'modelo_usado': modelo_path.name,
        'features_importantes': {
            'rsi_14': float(ultimo_dia['rsi_14']) if 'rsi_14' in ultimo_dia else None,
            'volatility_7': float(ultimo_dia['volatility_7']) if 'volatility_7' in ultimo_dia else None,
            'sma_7': float(ultimo_dia['sma_7']) if 'sma_7' in ultimo_dia else None,
        }
    }
    
    # Guardar predicción
    fecha_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    pred_file = PREDICTIONS_PATH / f"prediccion_{fecha_str}.json"
    
    with open(pred_file, 'w') as f:
        json.dump(resultado, f, indent=2)
    
    logging.info(f"💾 Predicción guardada: {pred_file}")
    
    # También guardar en historial
    historial_file = PREDICTIONS_PATH / "historial_predicciones.jsonl"
    with open(historial_file, 'a') as f:
        f.write(json.dumps(resultado) + '\n')
    
    # Pasar a siguiente task
    context['task_instance'].xcom_push(key='prediccion', value=resultado)
    
    return resultado


def enviar_notificacion(**context):
    """
    Envía notificación con la predicción (puede ser email, Slack, Telegram, etc.)
    Por ahora solo genera un reporte
    """
    logging.info("📧 Generando notificación...")
    
    prediccion = context['task_instance'].xcom_pull(
        task_ids='hacer_prediccion',
        key='prediccion'
    )
    
    # Crear mensaje
    emoji = "📈" if prediccion['prediccion'] == 1 else "📉"
    
    mensaje = f"""
        🤖 PREDICCIÓN DIARIA DE BITCOIN

        📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        💰 Precio actual: ${prediccion['precio_hoy']:,.2f}

        🔮 Predicción para MAÑANA:
        {emoji} {prediccion['direccion']}
        
        📊 Confianza: {prediccion['confianza']:.1%}

        📈 Probabilidades:
        • Sube: {prediccion['probabilidad_sube']:.1%}
        • Baja: {prediccion['probabilidad_baja']:.1%}

        🔍 Indicadores clave:
        • RSI-14: {prediccion['features_importantes'].get('rsi_14', 'N/A')}
        • Volatilidad 7d: {prediccion['features_importantes'].get('volatility_7', 'N/A')}
        • SMA-7: ${prediccion['features_importantes'].get('sma_7', 'N/A'):,.2f}

        🤖 Modelo: {prediccion['modelo_usado']}

        ⚠️ Disclaimer: Esto es una predicción estadística, no consejo financiero.
    """
    
    logging.info(mensaje)
    
    # Guardar mensaje
    notif_file = PREDICTIONS_PATH / f"notificacion_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(notif_file, 'w') as f:
        f.write(mensaje)
    
    logging.info(f"✅ Notificación guardada: {notif_file}")
    
    # Aquí podrías agregar:
    # - Envío por email (smtplib)
    # - Slack webhook
    # - Telegram bot
    # - Discord webhook
    # - SMS (Twilio)
    
    return str(notif_file)


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


# Definir argumentos por defecto
default_args = {
    'owner': 'Errodringer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Crear el DAG
with DAG(
    'bitcoin_daily_prediction',
    default_args=default_args,
    description='Pipeline de predicción diaria automatizada de Bitcoin',
    schedule_interval=None,
    catchup=False,
    tags=['bitcoin', 'ml', 'prediction', 'production'],
) as dag:
    
    # Task 1: Descargar datos recientes
    descargar_datos = PythonOperator(
        task_id='descargar_datos_recientes',
        python_callable=descargar_datos_recientes,
        provide_context=True,
    )
    
    # Task 2: Crear features
    crear_features = PythonOperator(
        task_id='crear_features',
        python_callable=crear_features_para_prediccion,
        provide_context=True,
    )
    
    # Task 3: Hacer predicción
    hacer_prediccion = PythonOperator(
        task_id='hacer_prediccion',
        python_callable=hacer_prediccion_hoy,
        provide_context=True,
    )
    
    # Task 4: Enviar notificación
    notificar = PythonOperator(
        task_id='enviar_notificacion',
        python_callable=enviar_notificacion,
        provide_context=True,
    )
    
    # Task 5: Generar reporte
    generar_reporte = PythonOperator(
        task_id='generar_reporte',
        python_callable=generar_reporte_diario,
        provide_context=True,
    )
    
    # Definir flujo
    descargar_datos >> crear_features >> hacer_prediccion >> [notificar, generar_reporte]
    # hacer_prediccion >> [notificar, generar_reporte]

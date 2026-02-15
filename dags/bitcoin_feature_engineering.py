"""
DAG de Feature Engineering para Bitcoin
Video 3: Procesamiento y preparación de datos para el modelo ML
Autor: Tu Canal de YouTube
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from pathlib import Path
import logging
import sys

from src.constants.constants import BASE_PATH, HISTORICAL_PATH, PROCESSED_PATH, FEATURES_PATH


# Verificar imports
try:
    import pandas as pd
    import numpy as np
    logging.info("✅ pandas y numpy importados correctamente")
except ImportError as e:
    logging.error(f"❌ Error importando librerías: {e}")
    raise

# Crear directorios
for path in [PROCESSED_PATH, FEATURES_PATH]:
    path.mkdir(parents=True, exist_ok=True)


def cargar_datos_historicos(**context):
    """
    Carga los datos históricos del pipeline anterior
    """
    logging.info("📂 Cargando datos históricos de Bitcoin...")
    
    # Buscar el archivo parquet más reciente
    archivos = sorted(HISTORICAL_PATH.glob("btc_historical_*.parquet"))
    
    if not archivos:
        raise FileNotFoundError(f"❌ No se encontraron archivos en {HISTORICAL_PATH}")
    
    archivo_mas_reciente = archivos[-1]
    logging.info(f"📁 Usando archivo: {archivo_mas_reciente.name}")
    
    # Cargar datos
    df = pd.read_parquet(archivo_mas_reciente)
    
    # Información básica
    logging.info(f"📊 Datos cargados: {len(df)} registros")
    logging.info(f"📅 Desde: {df['date'].min()} hasta {df['date'].max()}")
    logging.info(f"💰 Precio promedio: ${df['close'].mean():,.2f}")
    
    # Convertir fecha a datetime si no lo está
    df['date'] = pd.to_datetime(df['date'])
    
    # Ordenar por fecha
    df = df.sort_values('date').reset_index(drop=True)
    
    # Guardar en processed
    output_file = PROCESSED_PATH / "btc_raw.parquet"
    df.to_parquet(output_file, index=False)
    
    logging.info(f"✅ Datos guardados en: {output_file}")
    
    # Pasar metadata a siguiente task
    context['task_instance'].xcom_push(key='raw_file', value=str(output_file))
    context['task_instance'].xcom_push(key='num_records', value=len(df))
    
    return str(output_file)


def limpiar_datos(**context):
    """
    Limpieza de datos: eliminar duplicados, manejar valores faltantes
    """
    logging.info("🧹 Limpiando datos...")
    
    raw_file = context['task_instance'].xcom_pull(
        task_ids='cargar_datos',
        key='raw_file'
    )
    
    df = pd.read_parquet(raw_file)
    registros_iniciales = len(df)
    
    logging.info(f"📊 Registros iniciales: {registros_iniciales}")
    
    # 1. Eliminar duplicados
    duplicados_antes = df.duplicated(subset=['date']).sum()
    df = df.drop_duplicates(subset=['date'], keep='last')
    logging.info(f"🔍 Duplicados eliminados: {duplicados_antes}")
    
    # 2. Verificar valores nulos
    nulos_por_columna = df.isnull().sum()
    if nulos_por_columna.any():
        logging.warning("⚠️ Valores nulos encontrados:")
        for col, count in nulos_por_columna[nulos_por_columna > 0].items():
            logging.warning(f"   {col}: {count} nulos")
        
        # Rellenar con interpolación lineal
        df = df.interpolate(method='linear')
        logging.info("✅ Valores nulos rellenados con interpolación")
    
    # 3. Verificar orden cronológico
    df = df.sort_values('date').reset_index(drop=True)
    
    # 4. Verificar que no haya precios negativos o cero
    columnas_precio = ['open', 'high', 'low', 'close']
    for col in columnas_precio:
        valores_invalidos = (df[col] <= 0).sum()
        if valores_invalidos > 0:
            logging.warning(f"⚠️ {valores_invalidos} valores inválidos en {col}")
            # Reemplazar con el valor anterior válido
            df[col] = df[col].replace(0, np.nan).fillna(method='ffill')
    
    registros_finales = len(df)
    logging.info(f"📊 Registros finales: {registros_finales}")
    logging.info(f"🗑️ Registros eliminados: {registros_iniciales - registros_finales}")
    
    # Guardar datos limpios
    output_file = PROCESSED_PATH / "btc_clean.parquet"
    df.to_parquet(output_file, index=False)
    
    logging.info(f"✅ Datos limpios guardados en: {output_file}")
    
    context['task_instance'].xcom_push(key='clean_file', value=str(output_file))
    context['task_instance'].xcom_push(key='records_cleaned', value=registros_finales)
    
    return str(output_file)


def crear_features_tecnicas(**context):
    """
    Crear features técnicas: medias móviles, RSI, MACD, Bandas de Bollinger
    """
    logging.info("📈 Creando features técnicas...")
    
    clean_file = context['task_instance'].xcom_pull(
        task_ids='limpiar_datos',
        key='clean_file'
    )
    
    df = pd.read_parquet(clean_file)
    
    # ===== MEDIAS MÓVILES =====
    logging.info("📊 Calculando medias móviles...")
    
    # SMA - Simple Moving Averages
    periodos = [7, 14, 21, 50, 200]
    for periodo in periodos:
        df[f'sma_{periodo}'] = df['close'].rolling(window=periodo).mean()
        logging.info(f"   ✅ SMA-{periodo}")
    
    # EMA - Exponential Moving Averages
    for periodo in [12, 26]:
        df[f'ema_{periodo}'] = df['close'].ewm(span=periodo, adjust=False).mean()
        logging.info(f"   ✅ EMA-{periodo}")
    
    # ===== RSI - Relative Strength Index =====
    logging.info("📊 Calculando RSI...")
    
    def calcular_rsi(data, periodo=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['rsi_14'] = calcular_rsi(df['close'], 14)
    logging.info("   ✅ RSI-14")
    
    # ===== MACD - Moving Average Convergence Divergence =====
    logging.info("📊 Calculando MACD...")
    
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    logging.info("   ✅ MACD completo")
    
    # ===== BANDAS DE BOLLINGER =====
    logging.info("📊 Calculando Bandas de Bollinger...")
    
    periodo_bb = 20
    df['bb_middle'] = df['close'].rolling(window=periodo_bb).mean()
    bb_std = df['close'].rolling(window=periodo_bb).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    logging.info("   ✅ Bandas de Bollinger")
    
    # ===== VOLATILIDAD =====
    logging.info("📊 Calculando volatilidad...")
    
    # Volatilidad usando desviación estándar
    for periodo in [7, 14, 30]:
        df[f'volatility_{periodo}'] = df['close'].pct_change().rolling(window=periodo).std()
        logging.info(f"   ✅ Volatilidad-{periodo}")
    
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    logging.info("   ✅ ATR-14")
    
    # ===== VOLUMEN =====
    logging.info("📊 Creando features de volumen...")
    
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    logging.info("   ✅ Features de volumen")
    
    # Contar features creadas
    features_tecnicas = [col for col in df.columns if col not in 
                         ['date', 'open', 'high', 'low', 'close', 'volume', 'adj close']]
    
    logging.info(f"✅ Total de features técnicas creadas: {len(features_tecnicas)}")
    
    # Guardar
    output_file = PROCESSED_PATH / "btc_with_technical_features.parquet"
    df.to_parquet(output_file, index=False)
    
    logging.info(f"✅ Features técnicas guardadas en: {output_file}")
    
    context['task_instance'].xcom_push(key='technical_file', value=str(output_file))
    context['task_instance'].xcom_push(key='num_technical_features', value=len(features_tecnicas))
    
    return str(output_file)


def crear_features_temporales(**context):
    """
    Crear features basadas en tiempo: día de la semana, mes, trimestre, etc.
    """
    logging.info("📅 Creando features temporales...")
    
    technical_file = context['task_instance'].xcom_pull(
        task_ids='crear_features_tecnicas',
        key='technical_file'
    )
    
    df = pd.read_parquet(technical_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Features de tiempo
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Lunes, 6=Domingo
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    logging.info("   ✅ Día de semana, mes, trimestre")
    
    # Features cíclicas (útiles para ML)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    logging.info("   ✅ Features cíclicas")
    
    # ¿Es fin de semana?
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # ¿Es inicio/fin de mes?
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # ¿Es inicio/fin de trimestre?
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    
    logging.info("   ✅ Indicadores binarios")
    
    features_temporales = ['day_of_week', 'day_of_month', 'week_of_year', 'month', 
                          'quarter', 'year', 'day_sin', 'day_cos', 'month_sin', 
                          'month_cos', 'is_weekend', 'is_month_start', 'is_month_end',
                          'is_quarter_start', 'is_quarter_end']
    
    logging.info(f"✅ Total de features temporales creadas: {len(features_temporales)}")
    
    # Guardar
    output_file = PROCESSED_PATH / "btc_with_all_features.parquet"
    df.to_parquet(output_file, index=False)
    
    logging.info(f"✅ Features temporales guardadas en: {output_file}")
    
    context['task_instance'].xcom_push(key='all_features_file', value=str(output_file))
    context['task_instance'].xcom_push(key='num_temporal_features', value=len(features_temporales))
    
    return str(output_file)


def crear_target_variable(**context):
    """
    Crear la variable objetivo: predecir el precio de mañana
    """
    logging.info("🎯 Creando variable objetivo (target)...")
    
    all_features_file = context['task_instance'].xcom_pull(
        task_ids='crear_features_temporales',
        key='all_features_file'
    )
    
    df = pd.read_parquet(all_features_file)
    
    # Variable objetivo: precio de cierre del día siguiente
    df['target_next_close'] = df['close'].shift(-1)
    
    # Cambio porcentual del día siguiente
    df['target_pct_change'] = df['close'].pct_change(-1) * 100
    
    # Dirección: ¿Sube o baja? (clasificación binaria)
    df['target_direction'] = (df['target_next_close'] > df['close']).astype(int)
    
    logging.info("   ✅ target_next_close - Precio de mañana")
    logging.info("   ✅ target_pct_change - Cambio % de mañana")
    logging.info("   ✅ target_direction - ¿Sube (1) o baja (0)?")
    
    # Estadísticas del target
    logging.info(f"📊 Precio promedio siguiente día: ${df['target_next_close'].mean():,.2f}")
    logging.info(f"📊 Cambio % promedio: {df['target_pct_change'].mean():.2f}%")
    dias_sube = df['target_direction'].sum()
    dias_baja = len(df) - dias_sube
    logging.info(f"📈 Días que sube: {dias_sube} ({dias_sube/len(df)*100:.1f}%)")
    logging.info(f"📉 Días que baja: {dias_baja} ({dias_baja/len(df)*100:.1f}%)")
    
    # Guardar
    output_file = PROCESSED_PATH / "btc_with_target.parquet"
    df.to_parquet(output_file, index=False)
    
    logging.info(f"✅ Target variable guardada en: {output_file}")
    
    context['task_instance'].xcom_push(key='with_target_file', value=str(output_file))
    
    return str(output_file)


def preparar_dataset_final(**context):
    """
    Preparar el dataset final: eliminar NaNs, seleccionar features, split train/test
    """
    logging.info("🎁 Preparando dataset final...")
    
    with_target_file = context['task_instance'].xcom_pull(
        task_ids='crear_target_variable',
        key='with_target_file'
    )
    
    df = pd.read_parquet(with_target_file)
    
    registros_antes = len(df)
    logging.info(f"📊 Registros antes de limpieza: {registros_antes}")
    
    # Eliminar filas con NaN (principalmente al inicio por las ventanas móviles)
    df = df.dropna()
    
    registros_despues = len(df)
    logging.info(f"📊 Registros después de limpieza: {registros_despues}")
    logging.info(f"🗑️ Registros eliminados: {registros_antes - registros_despues}")
    
    # Seleccionar features para el modelo
    # Excluir: date, precio original, adj close, y las variables target (excepto una)
    columnas_excluir = ['date', 'open', 'high', 'low', 'close', 'volume', 
                        'adj close', 'target_next_close', 'target_pct_change']
    
    feature_columns = [col for col in df.columns if col not in columnas_excluir]
    
    logging.info(f"📋 Features seleccionadas: {len(feature_columns)}")
    
    # Crear dataset con features y target
    X = df[feature_columns].copy()
    y = df['target_direction'].copy()  # Usamos dirección como target
    
    # También guardamos los precios para referencia
    prices = df[['date', 'close', 'target_next_close']].copy()
    
    # Split temporal: 80% train, 20% test
    split_index = int(len(df) * 0.8)
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    prices_train = prices.iloc[:split_index]
    prices_test = prices.iloc[split_index:]
    
    logging.info(f"📊 Train set: {len(X_train)} registros")
    logging.info(f"📊 Test set: {len(X_test)} registros")
    logging.info(f"📅 Train periodo: {prices_train['date'].min()} a {prices_train['date'].max()}")
    logging.info(f"📅 Test periodo: {prices_test['date'].min()} a {prices_test['date'].max()}")
    
    # Guardar datasets
    X_train.to_parquet(FEATURES_PATH / "X_train.parquet", index=False)
    X_test.to_parquet(FEATURES_PATH / "X_test.parquet", index=False)
    y_train.to_frame('target').to_parquet(FEATURES_PATH / "y_train.parquet", index=False)
    y_test.to_frame('target').to_parquet(FEATURES_PATH / "y_test.parquet", index=False)
    prices_train.to_parquet(FEATURES_PATH / "prices_train.parquet", index=False)
    prices_test.to_parquet(FEATURES_PATH / "prices_test.parquet", index=False)
    
    # Guardar lista de features
    with open(FEATURES_PATH / "feature_names.txt", 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    
    logging.info(f"✅ Datasets guardados en: {FEATURES_PATH}")
    logging.info("📁 Archivos creados:")
    logging.info("   - X_train.parquet")
    logging.info("   - X_test.parquet")
    logging.info("   - y_train.parquet")
    logging.info("   - y_test.parquet")
    logging.info("   - prices_train.parquet")
    logging.info("   - prices_test.parquet")
    logging.info("   - feature_names.txt")
    
    # Metadata para siguiente pipeline
    context['task_instance'].xcom_push(key='num_features', value=len(feature_columns))
    context['task_instance'].xcom_push(key='train_size', value=len(X_train))
    context['task_instance'].xcom_push(key='test_size', value=len(X_test))
    
    return True


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
                <p><strong>Próximo paso:</strong> Video 4 - Entrenar modelo de predicción</p>
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


# Definir argumentos por defecto
default_args = {
    'owner': 'youtube_tutorial',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Crear el DAG
with DAG(
    'bitcoin_feature_engineering',
    default_args=default_args,
    description='Pipeline de feature engineering para Bitcoin ML',
    schedule_interval=None,  # Manual trigger (corre después del pipeline de datos)
    catchup=False,
    tags=['bitcoin', 'ml', 'feature-engineering'],
) as dag:
    
    # Task 1: Cargar datos
    cargar_datos = PythonOperator(
        task_id='cargar_datos',
        python_callable=cargar_datos_historicos,
        provide_context=True,
    )
    
    # Task 2: Limpiar datos
    limpiar = PythonOperator(
        task_id='limpiar_datos',
        python_callable=limpiar_datos,
        provide_context=True,
    )
    
    # Task 3: Features técnicas
    crear_tecnicas = PythonOperator(
        task_id='crear_features_tecnicas',
        python_callable=crear_features_tecnicas,
        provide_context=True,
    )
    
    # Task 4: Features temporales
    crear_temporales = PythonOperator(
        task_id='crear_features_temporales',
        python_callable=crear_features_temporales,
        provide_context=True,
    )
    
    # Task 5: Crear variable objetivo
    crear_target = PythonOperator(
        task_id='crear_target_variable',
        python_callable=crear_target_variable,
        provide_context=True,
    )
    
    # Task 6: Preparar dataset final
    preparar_dataset = PythonOperator(
        task_id='preparar_dataset_final',
        python_callable=preparar_dataset_final,
        provide_context=True,
    )
    
    # Task 7: Generar reporte
    generar_reporte = PythonOperator(
        task_id='generar_reporte',
        python_callable=generar_reporte_features,
        provide_context=True,
    )
    
    # Definir flujo
    cargar_datos >> limpiar >> crear_tecnicas >> crear_temporales >> crear_target >> preparar_dataset >> generar_reporte

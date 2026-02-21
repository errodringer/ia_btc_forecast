"""
API REST para servir predicciones de Bitcoin
"""
import sys
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import json

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = Flask(__name__)
CORS(app)  # Permitir CORS para requests desde el navegador
from src.constants.constants import MODELS_PATH, PREDICTIONS_PATH, FEATURES_PATH

# Cargar modelo y scaler al inicio
print("🚀 Cargando modelo...")
with open(MODELS_PATH / "random_forest.pkl", 'rb') as f:
    modelo = pickle.load(f)

with open(MODELS_PATH / "scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

with open(FEATURES_PATH / "feature_names.txt", 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

print("✅ Modelo cargado y listo")


def crear_features(df):
    """Crea las features necesarias para el modelo"""
    
    # Medias móviles
    # Usar los nombres correctos de columnas tras aplanar el MultiIndex
    close_col = 'close_btc-usd'
    high_col = 'high_btc-usd'
    low_col = 'low_btc-usd'
    open_col = 'open_btc-usd'
    volume_col = 'volume_btc-usd'

    for periodo in [7, 14, 21, 50, 200]:
        df[f'sma_{periodo}'] = df[close_col].rolling(window=periodo).mean()

    for periodo in [12, 26]:
        df[f'ema_{periodo}'] = df[close_col].ewm(span=periodo, adjust=False).mean()

    # RSI
    def calcular_rsi(data, periodo=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['rsi_14'] = calcular_rsi(df[close_col], 14)

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # Bandas de Bollinger
    periodo_bb = 20
    df['bb_middle'] = df[close_col].rolling(window=periodo_bb).mean()
    bb_std = df[close_col].rolling(window=periodo_bb).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']

    # Volatilidad
    for periodo in [7, 14, 30]:
        df[f'volatility_{periodo}'] = df[close_col].pct_change().rolling(window=periodo).std()

    # ATR
    high_low = df[high_col] - df[low_col]
    high_close = abs(df[high_col] - df[close_col].shift())
    low_close = abs(df[low_col] - df[close_col].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()

    # Volumen
    df['volume_sma_20'] = df[volume_col].rolling(window=20).mean()
    df['volume_ratio'] = df[volume_col] / df['volume_sma_20']
    
    # Precios históricos
    for dias_atras in [1, 2, 3]:
        df[f'close_lag_{dias_atras}'] = df[close_col].shift(dias_atras)
    
    # Diferencia absoluta de precio respecto a días anteriores
    for dias_atras in [1, 2, 3]:
        df[f'price_diff_{dias_atras}'] = df[close_col] - df[f'close_lag_{dias_atras}']
    
    # Cambio porcentual respecto a días anteriores
    for dias_atras in [1, 2, 3]:
        df[f'pct_change_{dias_atras}'] = (
            (df[close_col] - df[f'close_lag_{dias_atras}']) / 
            df[f'close_lag_{dias_atras}'] * 100
        )
    
    # Mínimo y máximo de los últimos 3, 7, 14 días
    for periodo in [3, 7, 14]:
        df[f'min_close_{periodo}d'] = df[close_col].rolling(window=periodo).min()
        df[f'max_close_{periodo}d'] = df[close_col].rolling(window=periodo).max()
        df[f'dist_to_min_{periodo}d'] = df[close_col] - df[f'min_close_{periodo}d']
        df[f'dist_to_max_{periodo}d'] = df[close_col] - df[f'max_close_{periodo}d']
    
    # Retorno de días anteriores
    for dias_atras in [1, 2, 3]:
        df[f'return_{dias_atras}d'] = (
            (df[close_col] - df[f'close_lag_{dias_atras}']) / 
            df[f'close_lag_{dias_atras}']
        )
    
    # Features temporales
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    
    return df


@app.route('/')
def home():
    """Endpoint de bienvenida"""
    return jsonify({
        'message': '🤖 Bitcoin ML Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Hacer predicción (requiere datos históricos)',
            '/predict/now': 'GET - Predicción para mañana (usa datos en vivo)',
            '/history': 'GET - Ver historial de predicciones',
            '/latest': 'GET - Ver última predicción guardada',
            '/health': 'GET - Health check'
        }
    })


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': modelo is not None,
        'scaler_loaded': scaler is not None
    })


@app.route('/predict/now', methods=['GET'])
def predict_now():
    """
    Descarga datos actuales y hace predicción para mañana
    """
    try:
        print("📥 Descargando datos actuales...")
        
        # Descargar últimos 200 días
        btc_data = yf.download("BTC-USD", period="200d", interval="1d", progress=False)
        
        if btc_data.empty:
            return jsonify({'error': 'No se pudieron descargar datos'}), 500
        
        # Preparar datos
        btc_data.reset_index(inplace=True)
        # Aplanar MultiIndex si existe
        if isinstance(btc_data.columns, pd.MultiIndex):
            btc_data.columns = ['_'.join([str(i) for i in col if i]).lower() for col in btc_data.columns.values]
        else:
            btc_data.columns = btc_data.columns.str.lower()
        # Renombrar columna de fecha a 'date'
        if 'date' not in btc_data.columns:
            # Buscar columna que contenga 'date'
            date_col = [col for col in btc_data.columns if 'date' in col]
            if date_col:
                btc_data.rename(columns={date_col[0]: 'date'}, inplace=True)
            else:
                btc_data['date'] = btc_data.index
        
        # Crear features
        df = crear_features(btc_data)
        
        # Tomar último día
        ultimo_dia = df.iloc[-1]
        fecha_hoy = ultimo_dia['date']
        precio_hoy = ultimo_dia['close_btc-usd']
        
        # Seleccionar features
        features = ultimo_dia[feature_names].values.reshape(1, -1)
        
        # Normalizar
        features_norm = scaler.transform(features)
        
        # Predecir
        prediccion = modelo.predict(features_norm)[0]
        probabilidades = modelo.predict_proba(features_norm)[0]
        
        resultado = {
            'timestamp': datetime.now().isoformat(),
            'fecha_hoy': fecha_hoy.isoformat(),
            'fecha_prediccion': (fecha_hoy + timedelta(days=1)).isoformat(),
            'precio_actual': float(precio_hoy),
            'prediccion': {
                'direccion': 'SUBE' if prediccion == 1 else 'BAJA',
                'valor': int(prediccion),
                'emoji': '📈' if prediccion == 1 else '📉'
            },
            'probabilidades': {
                'sube': float(probabilidades[1]),
                'baja': float(probabilidades[0])
            },
            'confianza': float(probabilidades[prediccion]),
            'indicadores': {
                'rsi_14': float(ultimo_dia['rsi_14']),
                'volatility_7': float(ultimo_dia['volatility_7']),
                'sma_7': float(ultimo_dia['sma_7'])
            }
        }
        
        print(f"✅ Predicción: {resultado['prediccion']['direccion']} ({resultado['confianza']:.1%})")
        
        return jsonify(resultado)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/latest', methods=['GET'])
def get_latest():
    """
    Devuelve la última predicción guardada
    """
    try:
        historial_file = PREDICTIONS_PATH / "historial_predicciones.jsonl"
        
        if not historial_file.exists():
            return jsonify({'error': 'No hay predicciones guardadas'}), 404
        
        # Leer última línea
        with open(historial_file, 'r') as f:
            lineas = f.readlines()
            if not lineas:
                return jsonify({'error': 'No hay predicciones guardadas'}), 404
            
            ultima = json.loads(lineas[-1])
        
        return jsonify(ultima)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def get_history():
    """
    Devuelve el historial de predicciones
    Query params:
    - limit: número de predicciones (default: 10)
    """
    try:
        limit = int(request.args.get('limit', 10))
        
        historial_file = PREDICTIONS_PATH / "historial_predicciones.jsonl"
        
        if not historial_file.exists():
            return jsonify({'error': 'No hay predicciones guardadas'}), 404
        
        with open(historial_file, 'r') as f:
            lineas = f.readlines()
        
        # Tomar las últimas N
        historial = [json.loads(line) for line in lineas[-limit:]]
        
        return jsonify({
            'total': len(lineas),
            'limit': limit,
            'predicciones': list(reversed(historial))  # Más reciente primero
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Estadísticas del modelo y predicciones
    """
    try:
        historial_file = PREDICTIONS_PATH / "historial_predicciones.jsonl"
        
        if not historial_file.exists():
            return jsonify({'error': 'No hay predicciones guardadas'}), 404
        
        with open(historial_file, 'r') as f:
            predicciones = [json.loads(line) for line in f.readlines()]
        
        # Calcular estadísticas
        total = len(predicciones)
        predicciones_sube = sum(1 for p in predicciones if p['prediccion'] == 1)
        predicciones_baja = total - predicciones_sube
        
        confianza_promedio = np.mean([p['confianza'] for p in predicciones])
        
        stats = {
            'total_predicciones': total,
            'predicciones_sube': predicciones_sube,
            'predicciones_baja': predicciones_baja,
            'porcentaje_sube': predicciones_sube / total * 100 if total > 0 else 0,
            'confianza_promedio': float(confianza_promedio),
            'primera_prediccion': predicciones[0]['fecha_prediccion'] if predicciones else None,
            'ultima_prediccion': predicciones[-1]['fecha_prediccion'] if predicciones else None
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*60)
    print("🚀 INICIANDO API DE PREDICCIÓN DE BITCOIN")
    print("="*60)
    print()
    print("📡 Endpoints disponibles:")
    print("   GET  /              - Información de la API")
    print("   GET  /health        - Health check")
    print("   GET  /predict/now   - Predicción actual")
    print("   GET  /latest        - Última predicción")
    print("   GET  /history       - Historial (limit=10)")
    print("   GET  /stats         - Estadísticas")
    print()
    print("🌐 La API correrá en: http://localhost:5050")
    print("="*60)
    print()
    
    # Correr la app
    app.run(host='0.0.0.0', port=5050, debug=True)

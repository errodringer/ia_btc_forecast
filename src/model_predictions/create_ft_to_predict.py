import logging

import pandas as pd
import numpy as np

from src.constants.constants import PREDICTIONS_PATH


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

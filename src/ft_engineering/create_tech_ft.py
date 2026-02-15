import logging
import pandas as pd

from src.constants.constants import PROCESSED_PATH


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


if __name__ == "__main__":
    # Prueba local
    crear_features_tecnicas()

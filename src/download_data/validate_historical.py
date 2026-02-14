import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import logging


from src.constants.constants import HISTORICAL_PATH

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def validar_datos_historicos(**context):
    """
    Valida que los datos históricos sean correctos
    Chequeos: gaps, valores negativos, outliers extremos
    """
    logging.info("🔍 Validando datos históricos...")
    
    # filepath = context['task_instance'].xcom_pull(
    #     task_ids='descargar_historicos',
    #     key='historical_file'
    # )
    filepath = "/Users/errodringer/Proyectos/ia_btc_forecast/data/historical/btc_historical_20260214.parquet"  # Para pruebas locales
    
    df = pd.read_parquet(filepath)
    
    errores = []
    
    # 1. Verificar que no haya valores negativos
    if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
        errores.append("❌ Valores negativos detectados")
    
    # 2. Verificar que high >= low
    if (df['high'] < df['low']).any():
        errores.append("❌ Precio alto menor que precio bajo detectado")
    
    # 3. Verificar gaps en las fechas (más de 2 días)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    date_diff = df['date'].diff()
    max_gap = date_diff.max().days if len(date_diff) > 0 else 0
    
    if max_gap > 3:
        logging.warning(f"⚠️ Gap máximo detectado: {max_gap} días")
    
    # 4. Verificar outliers (cambios mayores al 50% en un día)
    df['price_change'] = df['close'].pct_change().abs()
    outliers = df[df['price_change'] > 0.5]
    
    if len(outliers) > 0:
        logging.warning(f"⚠️ {len(outliers)} días con cambios extremos (>50%)")
    
    # 5. Verificar volumen
    if (df['volume'] == 0).sum() > 10:
        errores.append(f"❌ {(df['volume'] == 0).sum()} días sin volumen")
    
    if errores:
        error_msg = "\n".join(errores)
        logging.error(f"Errores de validación:\n{error_msg}")
        raise ValueError(f"Validación fallida: {error_msg}")
    
    logging.info("✅ Validación exitosa - Datos históricos son correctos")
    logging.info(f"📊 Registros validados: {len(df)}")
    logging.info(f"📅 Gap máximo: {max_gap} días")
    logging.info(f"📈 Días con cambios >50%: {len(outliers)}")
    
    return True

validar_datos_historicos()
import logging

import pandas as pd
import yfinance as yf

from src.constants.constants import PREDICTIONS_PATH


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

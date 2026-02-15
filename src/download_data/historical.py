import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import yfinance as yf
import pandas as pd
import logging

from datetime import datetime

from src.constants.constants import HISTORICAL_PATH


def descargar_datos_historicos(**context):
    try:
        # Descargar datos de los últimos 2 años
        ticker = "BTC-USD"
        btc_data = yf.download(
            ticker,
            period="2y",
            interval="1d",
            progress=False
        )

        if btc_data.empty:
            raise ValueError("❌ No se descargaron datos históricos")

        # Preparar datos
        btc_data.reset_index(inplace=True)
        # Aquí está el cambio clave - si tienes multiindex en columnas:
        if isinstance(btc_data.columns, pd.MultiIndex):
            btc_data.columns = [
                ' '.join(col).strip().split(' ', maxsplit=1)[0] 
                for col in btc_data.columns.values
            ]
        btc_data.columns = btc_data.columns.str.lower()

        # Guardar en formato parquet (más eficiente)
        filename = f"btc_historical_{datetime.now().strftime('%Y%m%d')}.parquet"
        filepath = HISTORICAL_PATH / filename
        btc_data.to_parquet(filepath, index=False)

        logging.info("🚀 Iniciando descarga de datos históricos de Bitcoin...")

        logging.info(f"✅ Datos históricos descargados: {len(btc_data)} registros")
        logging.info(f"📁 Guardado en: {filepath}")
        logging.info(f"📊 Rango de fechas: {btc_data['date'].min()} a {btc_data['date'].max()}")
        logging.info(f"💰 Precio más alto: ${btc_data['high'].max():.2f}")
        logging.info(f"💸 Precio más bajo: ${btc_data['low'].min():.2f}")

        # # Guardar metadata para siguiente task
        context['task_instance'].xcom_push(
            key='historical_file',
            value=str(filepath)
        )
        context['task_instance'].xcom_push(
            key='historical_records',
            value=len(btc_data)
        )
    except Exception as e:
        logging.error(f"❌ Error descargando datos históricos: {str(e)}")
        raise


if __name__ == "__main__":
    descargar_datos_historicos()

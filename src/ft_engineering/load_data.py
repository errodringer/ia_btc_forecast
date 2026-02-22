import logging
import pandas as pd

from src.constants.constants import HISTORICAL_PATH, PROCESSED_PATH


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

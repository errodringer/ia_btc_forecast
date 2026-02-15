"""
DAG de Feature Engineering para Bitcoin
Video 3: Procesamiento y preparación de datos para el modelo ML
Autor: Tu Canal de YouTube
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pathlib import Path
import logging
import sys

from src.constants.constants import PROCESSED_PATH, FEATURES_PATH


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


def limpiar_datos(**context):
    """
    Limpieza de datos: eliminar duplicados, manejar valores faltantes
    """
    logging.info("🧹 Limpiando datos...")

    # raw_file = context['task_instance'].xcom_pull(
    #     task_ids='cargar_datos',
    #     key='raw_file'
    # )
    raw_file = "/Users/errodringer/Proyectos/ia_btc_forecast/data/processed/btc_raw.parquet"

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

    # context['task_instance'].xcom_push(key='clean_file', value=str(output_file))
    # context['task_instance'].xcom_push(key='records_cleaned', value=registros_finales)

    return str(output_file)


if __name__ == "__main__":
    # Prueba local
    limpiar_datos()

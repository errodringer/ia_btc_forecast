import logging
import pandas as pd
import numpy as np

from src.constants.constants import PROCESSED_PATH


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

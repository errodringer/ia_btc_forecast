import logging
import pandas as pd

from src.constants.constants import PROCESSED_PATH


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

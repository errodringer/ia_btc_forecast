import logging


def validar_precio_actual(**context):
    """
    Valida que el precio actual sea razonable
    """
    logging.info("🔍 Validando precio actual...")

    current_price = context['task_instance'].xcom_pull(
        task_ids='download_data.descargar_precio_actual',
        key='current_price'
    )

    # Validaciones básicas
    if current_price <= 0:
        raise ValueError("❌ Precio actual es negativo o cero")

    # Verificar que esté en un rango razonable (entre $1,000 y $500,000)
    if current_price < 1000 or current_price > 500000:
        logging.warning(f"⚠️ Precio fuera del rango esperado: ${current_price:,.2f}")

    logging.info(f"✅ Precio actual validado: ${current_price:,.2f}")

    return True

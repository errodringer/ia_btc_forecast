import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging


from src.constants.constants import HISTORICAL_PATH

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def validar_precio_actual(**context):
    """
    Valida que el precio actual sea razonable
    """
    logging.info("🔍 Validando precio actual...")
    
    # current_price = context['task_instance'].xcom_pull(
    #     task_ids='descargar_precio_actual',
    #     key='current_price'
    # )
    current_price = 30000  # Para pruebas locales
    
    # Validaciones básicas
    if current_price <= 0:
        raise ValueError("❌ Precio actual es negativo o cero")
    
    # Verificar que esté en un rango razonable (entre $1,000 y $500,000)
    if current_price < 1000 or current_price > 500000:
        logging.warning(f"⚠️ Precio fuera del rango esperado: ${current_price:,.2f}")
    
    logging.info(f"✅ Precio actual validado: ${current_price:,.2f}")
    
    return True

validar_precio_actual()
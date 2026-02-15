import logging
import requests
import json

from datetime import datetime

from src.constants.constants import CURRENT_PATH


def descargar_precio_actual(**context):
    """
    Descarga el precio actual de Bitcoin desde CoinGecko (API gratuita)
    """
    logging.info("💎 Descargando precio actual de Bitcoin...")

    try:
        # CoinGecko API - No requiere API key
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': 'bitcoin',
            'vs_currencies': 'usd',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true'
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if 'bitcoin' not in data:
            raise ValueError("❌ No se recibió información de Bitcoin")

        btc_data = data['bitcoin']

        # Crear registro con timestamp
        current_data = {
            'timestamp': datetime.now(),
            'price_usd': btc_data['usd'],
            'volume_24h': btc_data.get('usd_24h_vol', 0),
            'change_24h': btc_data.get('usd_24h_change', 0),
            'last_updated': datetime.fromtimestamp(btc_data.get('last_updated_at', 0))
        }

        # Guardar en JSON
        filename = f"btc_current_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = CURRENT_PATH / filename

        with open(filepath, 'w') as f:
            json.dump(current_data, f, default=str, indent=2)

        logging.info(f"✅ Precio actual descargado: ${current_data['price_usd']:,.2f}")
        logging.info(f"📊 Volumen 24h: ${current_data['volume_24h']:,.0f}")
        logging.info(f"📈 Cambio 24h: {current_data['change_24h']:.2f}%")
        logging.info(f"📁 Guardado en: {filepath}")

        # Guardar para siguiente task
        context['task_instance'].xcom_push(
            key='current_file',
            value=str(filepath)
        )
        context['task_instance'].xcom_push(
            key='current_price',
            value=current_data['price_usd']
        )

        return str(filepath)

    except Exception as e:
        logging.error(f"❌ Error descargando precio actual: {str(e)}")
        raise


if __name__ == "__main__":
    descargar_precio_actual()

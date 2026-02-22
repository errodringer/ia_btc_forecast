import logging
from datetime import datetime

from src.constants.constants import PREDICTIONS_PATH


def enviar_notificacion(**context):
    """
    Envía notificación con la predicción (puede ser email, Slack, Telegram, etc.)
    Por ahora solo genera un reporte
    """
    logging.info("📧 Generando notificación...")
    
    prediccion = context['task_instance'].xcom_pull(
        task_ids='hacer_prediccion',
        key='prediccion'
    )
    
    # Crear mensaje
    emoji = "📈" if prediccion['prediccion'] == 1 else "📉"
    
    mensaje = f"""
        🤖 PREDICCIÓN DIARIA DE BITCOIN

        📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        💰 Precio actual: ${prediccion['precio_hoy']:,.2f}

        🔮 Predicción para MAÑANA:
        {emoji} {prediccion['direccion']}
        
        📊 Confianza: {prediccion['confianza']:.1%}

        📈 Probabilidades:
        • Sube: {prediccion['probabilidad_sube']:.1%}
        • Baja: {prediccion['probabilidad_baja']:.1%}

        🔍 Indicadores clave:
        • RSI-14: {prediccion['features_importantes'].get('rsi_14', 'N/A')}
        • Volatilidad 7d: {prediccion['features_importantes'].get('volatility_7', 'N/A')}
        • SMA-7: ${prediccion['features_importantes'].get('sma_7', 'N/A'):,.2f}

        🤖 Modelo: {prediccion['modelo_usado']}

        ⚠️ Disclaimer: Esto es una predicción estadística, no consejo financiero.
    """
    
    logging.info(mensaje)
    
    # Guardar mensaje
    notif_file = PREDICTIONS_PATH / f"notificacion_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(notif_file, 'w') as f:
        f.write(mensaje)
    
    logging.info(f"✅ Notificación guardada: {notif_file}")
    
    # Aquí podrías agregar:
    # - Envío por email (smtplib)
    # - Slack webhook
    # - Telegram bot
    # - Discord webhook
    # - SMS (Twilio)
    
    return str(notif_file)

import logging
import pickle
import json
from datetime import datetime, timedelta

import pandas as pd

from src.constants.constants import FEATURES_PATH, PREDICTIONS_PATH, MODELS_PATH


def hacer_prediccion_hoy(**context):
    """
    Hace la predicción para mañana usando el modelo entrenado
    """
    logging.info("🔮 Haciendo predicción para MAÑANA...")
    
    # Cargar datos con features
    features_file = context['task_instance'].xcom_pull(
        task_ids='crear_features',
        key='features_file'
    )
    
    # features_file = PROCESSED_PATH / "btc_with_all_features.parquet"
    df = pd.read_parquet(features_file)
    
    # Tomar el último día (hoy)
    ultimo_dia = df.iloc[-1].copy()
    fecha_hoy = ultimo_dia['date']
    precio_hoy = ultimo_dia['close']
    
    logging.info(f"📅 Fecha HOY: {fecha_hoy.strftime('%Y-%m-%d')}")
    logging.info(f"💰 Precio HOY: ${precio_hoy:,.2f}")
    
    # Cargar lista de features del entrenamiento
    with open(FEATURES_PATH / "feature_names.txt", 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Seleccionar solo las features que usó el modelo
    features_para_modelo = ultimo_dia[feature_names].values.reshape(1, -1)
    
    # Cargar scaler
    with open(MODELS_PATH / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Normalizar
    features_normalizadas = scaler.transform(features_para_modelo)
    
    # Cargar el mejor modelo
    modelo_path = MODELS_PATH / "best_model.pkl"
    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)
    
    logging.info(f"✅ Modelo cargado: {modelo_path.name}")
    
    # HACER PREDICCIÓN
    prediccion = modelo.predict(features_normalizadas)[0]
    probabilidades = modelo.predict_proba(features_normalizadas)[0]
    
    prob_baja = probabilidades[0]
    prob_sube = probabilidades[1]
    
    # Resultado
    direccion = "SUBE 📈" if prediccion == 1 else "BAJA 📉"
    confianza = prob_sube if prediccion == 1 else prob_baja
    
    logging.info(f"\n{'='*60}")
    logging.info(f"🔮 PREDICCIÓN PARA MAÑANA:")
    logging.info(f"{'='*60}")
    logging.info(f"   Dirección: {direccion}")
    logging.info(f"   Confianza: {confianza:.1%}")
    logging.info(f"   Probabilidad SUBE: {prob_sube:.1%}")
    logging.info(f"   Probabilidad BAJA: {prob_baja:.1%}")
    logging.info(f"{'='*60}\n")
    
    # Crear resultado
    resultado = {
        'fecha_prediccion': datetime.now().isoformat(),
        'fecha_hoy': fecha_hoy.isoformat(),
        'fecha_manana': (fecha_hoy + timedelta(days=1)).isoformat(),
        'precio_hoy': float(precio_hoy),
        'prediccion': int(prediccion),
        'direccion': direccion,
        'probabilidad_sube': float(prob_sube),
        'probabilidad_baja': float(prob_baja),
        'confianza': float(confianza),
        'modelo_usado': modelo_path.name,
        'features_importantes': {
            'rsi_14': float(ultimo_dia['rsi_14']) if 'rsi_14' in ultimo_dia else None,
            'volatility_7': float(ultimo_dia['volatility_7']) if 'volatility_7' in ultimo_dia else None,
            'sma_7': float(ultimo_dia['sma_7']) if 'sma_7' in ultimo_dia else None,
        }
    }
    
    # Guardar predicción
    fecha_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    pred_file = PREDICTIONS_PATH / f"prediccion_{fecha_str}.json"
    
    with open(pred_file, 'w') as f:
        json.dump(resultado, f, indent=2)
    
    logging.info(f"💾 Predicción guardada: {pred_file}")
    
    # También guardar en historial
    historial_file = PREDICTIONS_PATH / "historial_predicciones.jsonl"
    with open(historial_file, 'a') as f:
        f.write(json.dumps(resultado) + '\n')
    
    # Pasar a siguiente task
    context['task_instance'].xcom_push(key='prediccion', value=resultado)
    
    return resultado

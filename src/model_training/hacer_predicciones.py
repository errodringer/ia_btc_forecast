"""
Script para hacer predicciones con el modelo entrenado
Útil para demos y testing
"""
import sys
import pandas as pd
import pickle
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants.constants import FEATURES_PATH, MODELS_PATH



def cargar_mejor_modelo():
    """
    Carga el mejor modelo y el scaler
    """
    print("=" * 80)
    print("🤖 CARGANDO MODELO DE PREDICCIÓN")
    print("=" * 80)
    print()
    
    # Cargar scaler
    with open(MODELS_PATH / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler cargado")
    
    # Por defecto, usar Random Forest (suele ser el mejor)
    # Puedes cambiar a 'logistic_regression.pkl' o 'gradient_boosting.pkl'
    model_path = MODELS_PATH / "random_forest.pkl"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Modelo cargado: {model_path.name}")
    print()
    
    return model, scaler


def hacer_prediccion_individual(fecha_str=None):
    """
    Hace una predicción para una fecha específica del test set
    """
    print("=" * 80)
    print("🔮 HACIENDO PREDICCIÓN INDIVIDUAL")
    print("=" * 80)
    print()
    
    # Cargar modelo y scaler
    model, scaler = cargar_mejor_modelo()
    
    # Cargar datos de test
    X_test = pd.read_parquet(FEATURES_PATH / "X_test_scaled.parquet")
    y_test = pd.read_parquet(FEATURES_PATH / "y_test.parquet")['target']
    prices_test = pd.read_parquet(FEATURES_PATH / "prices_test.parquet")
    
    # Si no se especifica fecha, usar la última
    if fecha_str is None:
        idx = -1
        fecha = prices_test.iloc[idx]['date']
    else:
        # Buscar la fecha
        prices_test['date'] = pd.to_datetime(prices_test['date'])
        fecha = pd.to_datetime(fecha_str)
        idx = prices_test[prices_test['date'] == fecha].index[0]
    
    # Obtener features y precio
    features = X_test.iloc[idx]
    precio_actual = prices_test.iloc[idx]['close']
    precio_siguiente = prices_test.iloc[idx]['target_next_close']
    real = y_test.iloc[idx]
    
    # Hacer predicción
    prediccion = model.predict(features.values.reshape(1, -1))[0]
    probabilidad = model.predict_proba(features.values.reshape(1, -1))[0]
    
    # Mostrar resultados
    print(f"📅 Fecha: {fecha.strftime('%Y-%m-%d')}")
    print(f"💰 Precio actual: ${precio_actual:,.2f}")
    print(f"💰 Precio siguiente (real): ${precio_siguiente:,.2f}")
    print()
    
    print(f"🔮 PREDICCIÓN DEL MODELO:")
    print(f"   Predicción: {'📈 SUBE' if prediccion == 1 else '📉 BAJA'}")
    print(f"   Probabilidad de subida: {probabilidad[1]:.1%}")
    print(f"   Probabilidad de bajada: {probabilidad[0]:.1%}")
    print()
    
    print(f"✅ REALIDAD:")
    print(f"   Real: {'📈 SUBIÓ' if real == 1 else '📉 BAJÓ'}")
    cambio_real = ((precio_siguiente - precio_actual) / precio_actual) * 100
    print(f"   Cambio: {cambio_real:+.2f}%")
    print()
    
    # ¿Acertó?
    acerto = (prediccion == real)
    if acerto:
        print("🎉 ¡PREDICCIÓN CORRECTA!")
    else:
        print("❌ Predicción incorrecta")
    print()
    
    return {
        'fecha': fecha,
        'precio_actual': precio_actual,
        'precio_siguiente': precio_siguiente,
        'prediccion': int(prediccion),
        'real': int(real),
        'acerto': acerto,
        'probabilidad_subida': probabilidad[1],
        'cambio_real': cambio_real
    }


def evaluar_ultimos_n_dias(n=10):
    """
    Evalúa el modelo en los últimos N días del test set
    """
    print("=" * 80)
    print(f"📊 EVALUANDO ÚLTIMOS {n} DÍAS")
    print("=" * 80)
    print()
    
    # Cargar modelo
    model, scaler = cargar_mejor_modelo()
    
    # Cargar datos
    X_test = pd.read_parquet(FEATURES_PATH / "X_test_scaled.parquet")
    y_test = pd.read_parquet(FEATURES_PATH / "y_test.parquet")['target']
    prices_test = pd.read_parquet(FEATURES_PATH / "prices_test.parquet")
    
    # Obtener últimos n días
    ultimos_n = X_test.tail(n)
    y_real = y_test.tail(n)
    precios = prices_test.tail(n)
    
    # Hacer predicciones
    predicciones = model.predict(ultimos_n)
    probabilidades = model.predict_proba(ultimos_n)[:, 1]
    
    # Crear DataFrame de resultados
    resultados = pd.DataFrame({
        'fecha': precios['date'].values,
        'precio': precios['close'].values,
        'precio_siguiente': precios['target_next_close'].values,
        'prediccion': predicciones,
        'real': y_real.values,
        'prob_subida': probabilidades,
    })
    
    resultados['acerto'] = (resultados['prediccion'] == resultados['real'])
    resultados['cambio_pct'] = ((resultados['precio_siguiente'] - resultados['precio']) / resultados['precio']) * 100
    
    # Mostrar tabla
    print("📅 Fecha       | Precio    | Pred  | Real  | Prob  | Cambio | ✓")
    print("-" * 80)
    
    for _, row in resultados.iterrows():
        fecha = pd.to_datetime(row['fecha']).strftime('%Y-%m-%d')
        pred_emoji = '📈' if row['prediccion'] == 1 else '📉'
        real_emoji = '📈' if row['real'] == 1 else '📉'
        check = '✅' if row['acerto'] else '❌'
        
        print(f"{fecha} | ${row['precio']:8,.0f} | {pred_emoji}    | {real_emoji}    | {row['prob_subida']:.1%} | {row['cambio_pct']:+6.2f}% | {check}")
    
    print()
    
    # Estadísticas
    accuracy = resultados['acerto'].mean()
    print(f"📊 ESTADÍSTICAS:")
    print(f"   Accuracy: {accuracy:.1%}")
    print(f"   Predicciones correctas: {resultados['acerto'].sum()}/{len(resultados)}")
    print()
    
    # Análisis de ganancia hipotética
    print("💰 ANÁLISIS DE TRADING HIPOTÉTICO:")
    print("   (Si compramos cuando predice subida)")
    print()
    
    trades = resultados[resultados['prediccion'] == 1]
    if len(trades) > 0:
        ganancia_promedio = trades['cambio_pct'].mean()
        trades_ganadores = (trades['cambio_pct'] > 0).sum()
        
        print(f"   Número de trades: {len(trades)}")
        print(f"   Trades ganadores: {trades_ganadores} ({trades_ganadores/len(trades):.1%})")
        print(f"   Ganancia promedio por trade: {ganancia_promedio:+.2f}%")
        print(f"   Ganancia total acumulada: {trades['cambio_pct'].sum():+.2f}%")
    else:
        print("   No hubo señales de compra en este período")
    
    print()
    
    return resultados


def simular_trading_strategy():
    """
    Simula una estrategia de trading simple con el modelo
    """
    print("=" * 80)
    print("💰 SIMULACIÓN DE ESTRATEGIA DE TRADING")
    print("=" * 80)
    print()
    
    # Cargar modelo
    model, scaler = cargar_mejor_modelo()
    
    # Cargar datos de test
    X_test = pd.read_parquet(FEATURES_PATH / "X_test_scaled.parquet")
    y_test = pd.read_parquet(FEATURES_PATH / "y_test.parquet")['target']
    prices_test = pd.read_parquet(FEATURES_PATH / "prices_test.parquet")
    
    # Hacer predicciones
    predicciones = model.predict(X_test)
    probabilidades = model.predict_proba(X_test)[:, 1]
    
    # Estrategia: Comprar cuando predice subida CON alta confianza (>60%)
    capital_inicial = 1000  # $1000
    capital = capital_inicial
    posicion = 0  # 0 = sin posición, 1 = long
    trades = []
    
    print(f"💵 Capital inicial: ${capital_inicial:,.2f}")
    print(f"📊 Período: {prices_test['date'].min().strftime('%Y-%m-%d')} a {prices_test['date'].max().strftime('%Y-%m-%d')}")
    print()
    print("Estrategia: Comprar cuando probabilidad > 60%, vender al día siguiente")
    print()
    
    for i in range(len(predicciones) - 1):
        fecha = prices_test.iloc[i]['date']
        precio_actual = prices_test.iloc[i]['close']
        precio_siguiente = prices_test.iloc[i]['target_next_close']
        pred = predicciones[i]
        prob = probabilidades[i]
        
        # Si predice subida con confianza > 60%, comprar
        if pred == 1 and prob > 0.6 and posicion == 0:
            # Comprar
            posicion = 1
            precio_compra = precio_actual
            
            # Vender al día siguiente
            precio_venta = precio_siguiente
            ganancia_pct = ((precio_venta - precio_compra) / precio_compra) * 100
            ganancia_usd = capital * (ganancia_pct / 100)
            capital += ganancia_usd
            
            trades.append({
                'fecha_compra': fecha,
                'precio_compra': precio_compra,
                'precio_venta': precio_venta,
                'ganancia_pct': ganancia_pct,
                'capital': capital,
                'probabilidad': prob
            })
            
            posicion = 0  # Cerrar posición
    
    # Resultados
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        
        print(f"📊 RESULTADOS:")
        print(f"   Número de trades: {len(trades)}")
        print(f"   Trades ganadores: {(trades_df['ganancia_pct'] > 0).sum()}")
        print(f"   Trades perdedores: {(trades_df['ganancia_pct'] < 0).sum()}")
        print(f"   Win rate: {(trades_df['ganancia_pct'] > 0).mean():.1%}")
        print()
        
        ganancia_total = capital - capital_inicial
        retorno_pct = (ganancia_total / capital_inicial) * 100
        
        print(f"💰 CAPITAL FINAL: ${capital:,.2f}")
        print(f"💵 Ganancia total: ${ganancia_total:+,.2f} ({retorno_pct:+.2f}%)")
        print()
        
        # Comparar con buy & hold
        precio_inicial = prices_test.iloc[0]['close']
        precio_final = prices_test.iloc[-1]['close']
        buy_hold_return = ((precio_final - precio_inicial) / precio_inicial) * 100
        buy_hold_capital = capital_inicial * (1 + buy_hold_return / 100)
        
        print(f"📊 COMPARACIÓN CON BUY & HOLD:")
        print(f"   Retorno buy & hold: {buy_hold_return:+.2f}%")
        print(f"   Capital con buy & hold: ${buy_hold_capital:,.2f}")
        print()
        
        if retorno_pct > buy_hold_return:
            print(f"✅ La estrategia SUPERÓ al buy & hold por {retorno_pct - buy_hold_return:+.2f}%")
        else:
            print(f"❌ Buy & hold fue mejor por {buy_hold_return - retorno_pct:+.2f}%")
        
        print()
        print("⚠️ DISCLAIMER: Esto es una simulación con datos históricos.")
        print("   Los resultados pasados no garantizan resultados futuros.")
    else:
        print("❌ No se ejecutaron trades (ninguna señal con >60% confianza)")
    
    print()


if __name__ == "__main__":
    print("\n")
    
    # 1. Predicción individual
    print("\n🔮 EJEMPLO 1: PREDICCIÓN INDIVIDUAL\n")
    hacer_prediccion_individual()
    
    print("\n" + "="*80 + "\n")
    
    # 2. Últimos 10 días
    print("\n📊 EJEMPLO 2: ÚLTIMOS 10 DÍAS\n")
    evaluar_ultimos_n_dias(10)
    
    print("\n" + "="*80 + "\n")
    
    # 3. Simulación de trading
    print("\n💰 EJEMPLO 3: SIMULACIÓN DE TRADING\n")
    simular_trading_strategy()
    
    print("\n" + "="*80 + "\n")
    print("✅ ANÁLISIS COMPLETADO")
    print()

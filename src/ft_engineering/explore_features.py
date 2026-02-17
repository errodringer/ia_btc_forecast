"""
Script para visualizar y explorar las features creadas
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.constants.constants import PROCESSED_PATH, FEATURES_PATH


# Estilo de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def explorar_datos_procesados():
    """
    Explora los datos procesados y muestra estadísticas
    """
    print("=" * 80)
    print("📊 EXPLORANDO DATOS PROCESADOS")
    print("=" * 80)
    print()
    
    # Cargar datos con features
    df = pd.read_parquet(PROCESSED_PATH / "btc_with_target.parquet")
    
    print(f"📊 Shape del dataset: {df.shape}")
    print(f"📅 Periodo: {df['date'].min()} a {df['date'].max()}")
    print(f"📈 Días totales: {len(df)}")
    print()
    
    # Información de columnas
    print("📋 Columnas del dataset:")
    print("-" * 80)
    
    # Agrupar por tipo
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    technical_cols = [col for col in df.columns if any(x in col for x in ['sma', 'ema', 'rsi', 'macd', 'bb', 'volatility', 'atr', 'volume_'])]
    temporal_cols = [col for col in df.columns if any(x in col for x in ['day', 'week', 'month', 'quarter', 'year', 'is_', 'sin', 'cos'])]
    target_cols = [col for col in df.columns if 'target' in col]
    
    print(f"\n📊 Precios originales ({len(price_cols)}):")
    for col in price_cols:
        print(f"   - {col}")
    
    print(f"\n📈 Features técnicas ({len(technical_cols)}):")
    for col in technical_cols[:10]:  # Mostrar solo las primeras 10
        print(f"   - {col}")
    if len(technical_cols) > 10:
        print(f"   ... y {len(technical_cols) - 10} más")
    
    print(f"\n📅 Features temporales ({len(temporal_cols)}):")
    for col in temporal_cols:
        print(f"   - {col}")
    
    print(f"\n🎯 Variables objetivo ({len(target_cols)}):")
    for col in target_cols:
        print(f"   - {col}")
    
    print()
    
    # Estadísticas de las features técnicas
    print("=" * 80)
    print("📊 ESTADÍSTICAS DE FEATURES TÉCNICAS")
    print("=" * 80)
    print()
    
    # RSI
    print("💪 RSI (Relative Strength Index):")
    print(f"   Media: {df['rsi_14'].mean():.2f}")
    print(f"   Mínimo: {df['rsi_14'].min():.2f}")
    print(f"   Máximo: {df['rsi_14'].max():.2f}")
    print(f"   Sobrecomprado (>70): {(df['rsi_14'] > 70).sum()} días")
    print(f"   Sobrevendido (<30): {(df['rsi_14'] < 30).sum()} días")
    print()
    
    # MACD
    print("📊 MACD (Moving Average Convergence Divergence):")
    print(f"   MACD medio: {df['macd'].mean():.2f}")
    print(f"   Señales alcistas (MACD > Signal): {(df['macd'] > df['macd_signal']).sum()} días")
    print(f"   Señales bajistas (MACD < Signal): {(df['macd'] < df['macd_signal']).sum()} días")
    print()
    
    # Volatilidad
    print("📏 Volatilidad:")
    print(f"   Volatilidad 7 días (media): {df['volatility_7'].mean():.4f}")
    print(f"   Volatilidad 14 días (media): {df['volatility_14'].mean():.4f}")
    print(f"   Volatilidad 30 días (media): {df['volatility_30'].mean():.4f}")
    print()
    
    # Estadísticas del target
    print("=" * 80)
    print("🎯 ESTADÍSTICAS DE LA VARIABLE OBJETIVO")
    print("=" * 80)
    print()
    
    print("🎯 Target Direction (¿Sube o baja?):")
    dias_sube = df['target_direction'].sum()
    dias_baja = len(df) - dias_sube - df['target_direction'].isna().sum()
    print(f"   Días que sube (1): {dias_sube} ({dias_sube/len(df)*100:.1f}%)")
    print(f"   Días que baja (0): {dias_baja} ({dias_baja/len(df)*100:.1f}%)")
    print()
    
    print("📊 Target Price Change (%):")
    print(f"   Media: {df['target_pct_change'].mean():.2f}%")
    print(f"   Mediana: {df['target_pct_change'].median():.2f}%")
    print(f"   Std: {df['target_pct_change'].std():.2f}%")
    print(f"   Cambio máximo: {df['target_pct_change'].max():.2f}%")
    print(f"   Cambio mínimo: {df['target_pct_change'].min():.2f}%")
    print()


def explorar_datasets_finales():
    """
    Explora los datasets de train y test
    """
    print("=" * 80)
    print("🎁 EXPLORANDO DATASETS FINALES (TRAIN/TEST)")
    print("=" * 80)
    print()
    
    # Cargar datasets
    X_train = pd.read_parquet(FEATURES_PATH / "X_train.parquet")
    X_test = pd.read_parquet(FEATURES_PATH / "X_test.parquet")
    y_train = pd.read_parquet(FEATURES_PATH / "y_train.parquet")
    y_test = pd.read_parquet(FEATURES_PATH / "y_test.parquet")
    prices_train = pd.read_parquet(FEATURES_PATH / "prices_train.parquet")
    prices_test = pd.read_parquet(FEATURES_PATH / "prices_test.parquet")
    
    print("📊 TRAIN SET:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   Periodo: {prices_train['date'].min()} a {prices_train['date'].max()}")
    print(f"   Precio promedio: ${prices_train['close'].mean():,.2f}")
    print(f"   Balance de clases:")
    print(f"      Sube: {y_train['target'].sum()} ({y_train['target'].mean()*100:.1f}%)")
    print(f"      Baja: {len(y_train) - y_train['target'].sum()} ({(1-y_train['target'].mean())*100:.1f}%)")
    print()
    
    print("📊 TEST SET:")
    print(f"   X_test shape: {X_test.shape}")
    print(f"   y_test shape: {y_test.shape}")
    print(f"   Periodo: {prices_test['date'].min()} a {prices_test['date'].max()}")
    print(f"   Precio promedio: ${prices_test['close'].mean():,.2f}")
    print(f"   Balance de clases:")
    print(f"      Sube: {y_test['target'].sum()} ({y_test['target'].mean()*100:.1f}%)")
    print(f"      Baja: {len(y_test) - y_test['target'].sum()} ({(1-y_test['target'].mean())*100:.1f}%)")
    print()
    
    # Lista de features
    print("📋 FEATURES EN EL MODELO:")
    print(f"   Total: {len(X_train.columns)} features")
    print()
    
    with open(FEATURES_PATH / "feature_names.txt", 'r') as f:
        features = f.read().strip().split('\n')
    
    print("   Top 20 features:")
    for i, feature in enumerate(features[:20], 1):
        print(f"      {i}. {feature}")
    
    if len(features) > 20:
        print(f"   ... y {len(features) - 20} más")
    print()
    
    # Estadísticas de features
    print("📊 ESTADÍSTICAS DE FEATURES (TRAIN):")
    print()
    print(X_train.describe().T[['mean', 'std', 'min', 'max']].head(15))
    print()


def mostrar_correlaciones_importantes():
    """
    Muestra las correlaciones más importantes con el target
    """
    print("=" * 80)
    print("🔗 CORRELACIONES CON EL TARGET")
    print("=" * 80)
    print()
    
    # Cargar datos
    df = pd.read_parquet(PROCESSED_PATH / "btc_with_target.parquet")
    
    # Seleccionar solo features numéricas (excluir date)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calcular correlaciones con target_direction
    correlations = df[numeric_cols].corrwith(df['target_direction']).sort_values(ascending=False)
    
    print("🔝 TOP 15 FEATURES MÁS CORRELACIONADAS (positivamente):")
    print("-" * 80)
    for i, (feature, corr) in enumerate(correlations.head(15).items(), 1):
        if feature != 'target_direction':
            print(f"{i:2}. {feature:30} → {corr:+.4f}")
    print()
    
    print("🔻 TOP 15 FEATURES MÁS CORRELACIONADAS (negativamente):")
    print("-" * 80)
    for i, (feature, corr) in enumerate(correlations.tail(15).items(), 1):
        if feature not in ['target_direction', 'target_next_close', 'target_pct_change']:
            print(f"{i:2}. {feature:30} → {corr:+.4f}")
    print()


def verificar_calidad_datos():
    """
    Verifica la calidad de los datos procesados
    """
    print("=" * 80)
    print("✅ VERIFICACIÓN DE CALIDAD DE DATOS")
    print("=" * 80)
    print()
    
    # Cargar train set
    X_train = pd.read_parquet(FEATURES_PATH / "X_train.parquet")
    y_train = pd.read_parquet(FEATURES_PATH / "y_train.parquet")
    
    # 1. Verificar NaNs
    nans_count = X_train.isna().sum().sum()
    print(f"1️⃣ Valores NaN en X_train: {nans_count}")
    if nans_count == 0:
        print("   ✅ Sin valores NaN")
    else:
        print(f"   ⚠️ Encontrados {nans_count} NaNs")
        print("   Columnas con NaN:")
        for col in X_train.columns:
            nan_count = X_train[col].isna().sum()
            if nan_count > 0:
                print(f"      - {col}: {nan_count}")
    print()
    
    # 2. Verificar infinitos
    inf_count = np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()
    print(f"2️⃣ Valores infinitos en X_train: {inf_count}")
    if inf_count == 0:
        print("   ✅ Sin valores infinitos")
    else:
        print(f"   ⚠️ Encontrados {inf_count} infinitos")
    print()
    
    # 3. Verificar duplicados
    duplicados = X_train.duplicated().sum()
    print(f"3️⃣ Filas duplicadas: {duplicados}")
    if duplicados == 0:
        print("   ✅ Sin duplicados")
    else:
        print(f"   ⚠️ Encontrados {duplicados} duplicados")
    print()
    
    # 4. Verificar balance de clases
    balance = y_train['target'].value_counts()
    print(f"4️⃣ Balance de clases:")
    print(f"   Clase 0 (baja): {balance[0]} ({balance[0]/len(y_train)*100:.1f}%)")
    print(f"   Clase 1 (sube): {balance[1]} ({balance[1]/len(y_train)*100:.1f}%)")
    ratio = min(balance[0], balance[1]) / max(balance[0], balance[1])
    if ratio > 0.4:
        print(f"   ✅ Balance aceptable (ratio: {ratio:.2f})")
    else:
        print(f"   ⚠️ Desbalance significativo (ratio: {ratio:.2f})")
    print()
    
    # 5. Verificar rangos de features
    print("5️⃣ Rangos de features (primeras 10):")
    for col in X_train.columns[:10]:
        min_val = X_train[col].min()
        max_val = X_train[col].max()
        mean_val = X_train[col].mean()
        print(f"   {col:30} → [{min_val:10.2f}, {max_val:10.2f}] (mean: {mean_val:10.2f})")
    print()
    
    print("=" * 80)
    print("✅ VERIFICACIÓN COMPLETADA")
    print("=" * 80)
    print()


if __name__ == "__main__":
    # Ejecutar todas las exploraciones
    explorar_datos_procesados()
    print("\n" + "="*80 + "\n")
    
    explorar_datasets_finales()
    print("\n" + "="*80 + "\n")
    
    mostrar_correlaciones_importantes()
    print("\n" + "="*80 + "\n")
    
    verificar_calidad_datos()
    
    print("=" * 80)
    print("🎉 EXPLORACIÓN COMPLETADA")
    print("=" * 80)
    print()
    print("💡 Próximos pasos:")
    print("   1. Revisar las correlaciones más importantes")
    print("   2. Identificar features redundantes")
    print("   3. Entrenar el modelo de ML")
    print()

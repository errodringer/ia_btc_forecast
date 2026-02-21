"""
Dashboard interactivo para visualizar predicciones de Bitcoin
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime, timedelta
import pickle

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.constants.constants import MODELS_PATH, PREDICTIONS_PATH, FEATURES_PATH


# Configuración de la página
st.set_page_config(
    page_title="Bitcoin ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Configuración de rutas
# BASE_PATH = Path("/opt/airflow/data")
# MODELS_PATH = BASE_PATH / "models"
# PREDICTIONS_PATH = BASE_PATH / "predictions"

# Título
st.title("🤖 Bitcoin ML Prediction Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuración")

# Funciones auxiliares
@st.cache_data(ttl=300)  # Cache por 5 minutos
def cargar_historial():
    """Carga el historial de predicciones"""
    historial_file = PREDICTIONS_PATH / "historial_predicciones.jsonl"
    
    if not historial_file.exists():
        return pd.DataFrame()
    
    with open(historial_file, 'r') as f:
        predicciones = [json.loads(line) for line in f.readlines()]
    
    df = pd.DataFrame(predicciones)
    df['fecha_hoy'] = pd.to_datetime(df['fecha_hoy'])
    df['fecha_prediccion'] = pd.to_datetime(df['fecha_prediccion'])
    
    return df


@st.cache_data(ttl=300)
def cargar_metricas_modelo():
    """Carga las métricas del modelo"""
    results_file = MODELS_PATH / "test_results.json"
    
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        resultados = json.load(f)
    
    return resultados


def mostrar_ultima_prediccion():
    """Muestra la última predicción en grande"""
    df = cargar_historial()
    
    if df.empty:
        st.warning("⚠️ No hay predicciones guardadas aún")
        return
    
    ultima = df.iloc[-1]
    
    # Determinar color
    color = "green" if ultima['prediccion'] == 1 else "red"
    emoji = "📈" if ultima['prediccion'] == 1 else "📉"
    
    # Crear columnas
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.metric(
            label="💰 Precio Actual",
            value=f"${ultima['precio_hoy']:,.2f}",
            delta=None
        )
    
    with col2:
        st.markdown(f"### {emoji} {ultima['direccion']}")
        st.progress(float(ultima['confianza']))
        st.caption(f"Confianza: {ultima['confianza']:.1%}")
    
    with col3:
        st.metric(
            label="📅 Fecha",
            value=ultima['fecha_hoy'].strftime('%Y-%m-%d')
        )
    
    # Probabilidades
    st.markdown("#### 📊 Probabilidades")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="📈 Probabilidad Sube",
            value=f"{ultima['probabilidad_sube']:.1%}"
        )
    
    with col2:
        st.metric(
            label="📉 Probabilidad Baja",
            value=f"{ultima['probabilidad_baja']:.1%}"
        )


def mostrar_grafico_historial():
    """Muestra gráfico del historial de predicciones"""
    df = cargar_historial()
    
    if df.empty:
        st.warning("⚠️ No hay suficientes datos para el gráfico")
        return
    
    # Crear gráfico
    fig = go.Figure()
    
    # Línea de precio
    fig.add_trace(go.Scatter(
        x=df['fecha_hoy'],
        y=df['precio_hoy'],
        mode='lines+markers',
        name='Precio',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Marcar predicciones correctas/incorrectas (si tuviéramos la info)
    # Por ahora solo mostramos las predicciones
    
    # Colorear según predicción
    colors = ['green' if p == 1 else 'red' for p in df['prediccion']]
    
    fig.add_trace(go.Scatter(
        x=df['fecha_hoy'],
        y=df['precio_hoy'],
        mode='markers',
        name='Predicción',
        marker=dict(
            size=12,
            color=colors,
            symbol='triangle-up',
            line=dict(color='white', width=1)
        ),
        text=[f"{'SUBE' if p == 1 else 'BAJA'}<br>Confianza: {c:.1%}" 
              for p, c in zip(df['prediccion'], df['confianza'])],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="📈 Historial de Predicciones y Precios",
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def mostrar_distribucion_predicciones():
    """Muestra distribución de predicciones"""
    df = cargar_historial()
    
    if df.empty:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart de predicciones
        counts = df['prediccion'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['SUBE', 'BAJA'],
            values=[counts.get(1, 0), counts.get(0, 0)],
            marker=dict(colors=['green', 'red']),
            hole=0.4
        )])
        
        fig.update_layout(
            title="🎯 Distribución de Predicciones",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Histograma de confianza
        fig = px.histogram(
            df,
            x='confianza',
            nbins=20,
            title='📊 Distribución de Confianza',
            labels={'confianza': 'Confianza'},
            color_discrete_sequence=['#667eea']
        )
        
        fig.update_layout(height=300)
        
        st.plotly_chart(fig, use_container_width=True)


def mostrar_metricas_modelo():
    """Muestra métricas del modelo"""
    resultados = cargar_metricas_modelo()
    
    if not resultados:
        st.warning("⚠️ No se encontraron métricas del modelo")
        return
    
    st.markdown("### 🤖 Métricas del Modelo")
    
    # Encontrar mejor modelo
    mejor_modelo = max(resultados, key=lambda x: x['f1'])
    
    st.info(f"🏆 Mejor modelo: **{mejor_modelo['model']}**")
    
    # Mostrar métricas en columnas
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{mejor_modelo['accuracy']:.1%}")
    
    with col2:
        st.metric("Precision", f"{mejor_modelo['precision']:.1%}")
    
    with col3:
        st.metric("Recall", f"{mejor_modelo['recall']:.1%}")
    
    with col4:
        st.metric("F1-Score", f"{mejor_modelo['f1']:.3f}")
    
    with col5:
        st.metric("ROC-AUC", f"{mejor_modelo['roc_auc']:.3f}")
    
    # Tabla comparativa
    st.markdown("#### Comparación de Modelos")
    
    df_modelos = pd.DataFrame(resultados)
    df_modelos = df_modelos[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
    df_modelos.columns = ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    
    # Destacar mejor modelo
    def highlight_best(row):
        if row['Modelo'] == mejor_modelo['model']:
            return ['background-color: #d4edda'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        df_modelos.style.apply(highlight_best, axis=1).format({
            'Accuracy': '{:.1%}',
            'Precision': '{:.1%}',
            'Recall': '{:.1%}',
            'F1': '{:.3f}',
            'ROC-AUC': '{:.3f}'
        }),
        use_container_width=True
    )


def mostrar_estadisticas():
    """Muestra estadísticas generales"""
    df = cargar_historial()
    
    if df.empty:
        return
    
    st.markdown("### 📊 Estadísticas Generales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📝 Total Predicciones",
            len(df)
        )
    
    with col2:
        predicciones_sube = (df['prediccion'] == 1).sum()
        st.metric(
            "📈 Predicciones SUBE",
            f"{predicciones_sube} ({predicciones_sube/len(df)*100:.1f}%)"
        )
    
    with col3:
        predicciones_baja = (df['prediccion'] == 0).sum()
        st.metric(
            "📉 Predicciones BAJA",
            f"{predicciones_baja} ({predicciones_baja/len(df)*100:.1f}%)"
        )
    
    with col4:
        confianza_promedio = df['confianza'].mean()
        st.metric(
            "🎯 Confianza Promedio",
            f"{confianza_promedio:.1%}"
        )


def mostrar_tabla_predicciones():
    """Muestra tabla detallada de predicciones"""
    df = cargar_historial()
    
    if df.empty:
        return
    
    st.markdown("### 📋 Historial Detallado")
    
    # Preparar datos para mostrar
    df_display = df.copy()
    df_display['Fecha'] = df_display['fecha_hoy'].dt.strftime('%Y-%m-%d')
    df_display['Precio'] = df_display['precio_hoy'].apply(lambda x: f"${x:,.2f}")
    df_display['Predicción'] = df_display['direccion']
    df_display['Confianza'] = df_display['confianza'].apply(lambda x: f"{x:.1%}")
    df_display['Prob. Sube'] = df_display['probabilidad_sube'].apply(lambda x: f"{x:.1%}")
    
    # Seleccionar columnas
    df_display = df_display[['Fecha', 'Precio', 'Predicción', 'Confianza', 'Prob. Sube']]
    
    # Mostrar las últimas 20
    st.dataframe(
        df_display.tail(20).iloc[::-1],  # Invertir para mostrar más reciente primero
        use_container_width=True,
        hide_index=True
    )


# MAIN APP
def main():
    # Menú del sidebar
    opcion = st.sidebar.radio(
        "📍 Navegación",
        ["🏠 Dashboard", "📊 Métricas del Modelo", "📋 Historial Completo"]
    )
    
    # Botón de refresh
    if st.sidebar.button("🔄 Actualizar Datos"):
        st.cache_data.clear()
        st.rerun()
    
    # Mostrar última actualización
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Contenido según opción
    if opcion == "🏠 Dashboard":
        # Dashboard principal
        st.header("🏠 Dashboard Principal")
        
        # Última predicción
        st.markdown("### 🔮 Última Predicción")
        mostrar_ultima_prediccion()
        
        st.markdown("---")
        
        # Gráfico de historial
        mostrar_grafico_historial()
        
        st.markdown("---")
        
        # Distribuciones
        mostrar_distribucion_predicciones()
        
        st.markdown("---")
        
        # Estadísticas
        mostrar_estadisticas()
    
    elif opcion == "📊 Métricas del Modelo":
        # Métricas del modelo
        st.header("📊 Métricas del Modelo")
        mostrar_metricas_modelo()
    
    elif opcion == "📋 Historial Completo":
        # Historial completo
        st.header("📋 Historial Completo de Predicciones")
        mostrar_tabla_predicciones()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🤖 Bitcoin ML Prediction System | Powered by Airflow + Scikit-learn + Streamlit</p>
            <p style='font-size: 12px; color: gray;'>
                ⚠️ Disclaimer: Estas predicciones son generadas por un modelo de ML. 
                No constituyen consejo financiero. Invierte bajo tu propio riesgo.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

# 🤖 Sistema Completo de Predicción de Bitcoin con ML

## 📺 Serie Completa de YouTube - Machine Learning en Producción

Un sistema end-to-end de Machine Learning para predecir el precio de Bitcoin, automatizado con Apache Airflow.

---

## 🎯 ¿Qué hace este sistema?

Este proyecto implementa un pipeline completo de ML que:

1. 📥 **Descarga datos** de Bitcoin automáticamente
2. 🔧 **Crea 40+ features** técnicas (RSI, MACD, Bollinger, etc.)
3. 🤖 **Entrena modelos** de ML (Random Forest, Gradient Boosting, etc.)
4. 🔮 **Predice** el precio de mañana diariamente
5. 📊 **Visualiza** resultados en dashboard interactivo
6. 🌐 **Expone API** REST para consultas

**Todo automatizado y corriendo en producción con Airflow.**

---

## 📹 Videos de la Serie

| Video | Tema | DAG |
|-------|------|-----|
| 1 | Setup de Airflow | - |
| 2 | Pipeline de Datos | `bitcoin_data_pipeline` |
| 3 | Feature Engineering | `bitcoin_feature_engineering` |
| 4 | Entrenamiento ML | `bitcoin_model_training` |
| 5 | Deployment y API | `bitcoin_daily_prediction` |

---

## 🔄 Arquitectura del Sistema

### Flujo de DAGs Encadenados

```
EJECUCIÓN INICIAL (Setup)
├─────────────────────────────────────────┐
│                                         │
│  1. bitcoin_data_pipeline              │
│     └─> Descarga datos históricos      │
│         └─> Guarda en /historical/     │
│                                         │
│  TRIGGER AUTOMÁTICO ↓                  │
│                                         │
│  2. bitcoin_feature_engineering        │
│     └─> Carga datos históricos         │
│     └─> Crea 40+ features             │
│     └─> Guarda X_train, X_test         │
│                                         │
│  TRIGGER AUTOMÁTICO ↓                  │
│                                         │
│  3. bitcoin_model_training             │
│     └─> Carga features                 │
│     └─> Entrena 3 modelos              │
│     └─> Evalúa en test set             │
│     └─> Guarda mejor modelo            │
│                                         │
└─────────────────────────────────────────┘

EJECUCIÓN DIARIA (Producción)
┌─────────────────────────────────────────┐
│                                         │
│  4. bitcoin_daily_prediction           │
│     └─> Corre TODOS LOS DÍAS 9 AM      │
│     └─> Descarga datos frescos         │
│     └─> Crea features                  │
│     └─> Usa modelo entrenado           │
│     └─> Predice para mañana            │
│     └─> Genera reporte HTML            │
│     └─> Actualiza historial            │
│                                         │
└─────────────────────────────────────────┘

CONSULTAS EN TIEMPO REAL
┌─────────────────────────────────────────┐
│                                         │
│  API REST (Flask)                       │
│     └─> GET /predict/now               │
│     └─> GET /latest                    │
│     └─> GET /history                   │
│                                         │
│  Dashboard (Streamlit)                  │
│     └─> Visualización interactiva      │
│     └─> Gráficos de predicciones       │
│     └─> Métricas del modelo            │
│                                         │
└─────────────────────────────────────────┘
```

---

## 📦 Estructura del Proyecto

```
bitcoin-ml-airflow/
├── dags/
│   ├── bitcoin_data_pipeline.py         # Video 2: Descarga datos
│   ├── bitcoin_feature_engineering.py   # Video 3: Crea features
│   ├── bitcoin_model_training.py        # Video 4: Entrena modelos
│   └── bitcoin_daily_prediction.py      # Video 5: Predicción diaria
│
├── api/
│   └── api_prediccion.py                # API REST con Flask
│
├── dashboard/
│   └── dashboard_streamlit.py           # Dashboard interactivo
│
├── scripts/
│   ├── hacer_predicciones.py            # Script de prueba
│   ├── explorar_features.py             # Análisis de features
│   ├── test_pipeline.py                 # Test sin Airflow
│   └── diagnostico.py                   # Diagnóstico del sistema
│
├── requirements.txt                      # Dependencias Python
└── README.md                            # Este archivo
```

---

## 🚀 Quick Start

### 1. Requisitos Previos

- Docker y Docker Compose instalados
- Python 3.12
- 8GB RAM mínimo
- 10GB espacio en disco

### 2. Instalación de Airflow

```bash
# Descargar docker-compose de Airflow
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.3/docker-compose.yaml'

# Crear directorios necesarios
mkdir -p ./dags ./logs ./plugins ./data

# Inicializar Airflow
docker-compose up airflow-init

# Iniciar Airflow
docker-compose up -d
```

### 3. Instalar Dependencias en el Contenedor

```bash
# Identificar el contenedor de worker
docker ps

# Instalar dependencias
docker exec -it <airflow-worker-container-id> pip install \
    yfinance pandas numpy scikit-learn matplotlib seaborn \
    flask flask-cors streamlit plotly --break-system-packages
```

O usar el archivo requirements:

```bash
docker cp requirements.txt <container-id>:/tmp/
docker exec -it <container-id> pip install -r /tmp/requirements.txt --break-system-packages
```

### 4. Copiar los DAGs

```bash
# Copiar todos los DAGs a la carpeta de Airflow
cp dags/*.py ~/airflow/dags/
```

### 5. Ejecutar el Pipeline Inicial

1. Abre Airflow UI: http://localhost:8080
   - Usuario: `airflow`
   - Password: `airflow`

2. Activa los DAGs en este orden:
   - ✅ `bitcoin_data_pipeline`
   - ✅ `bitcoin_feature_engineering`
   - ✅ `bitcoin_model_training`
   - ✅ `bitcoin_daily_prediction`

3. Ejecuta manualmente `bitcoin_data_pipeline`
   - Esto disparará automáticamente los siguientes

4. Espera unos minutos para que termine todo el pipeline inicial

---

## 📊 Estructura de Datos

### Directorios Generados

```
/opt/airflow/data/
├── historical/                          # Datos descargados
│   └── btc_historical_YYYYMMDD.parquet
│
├── processed/                           # Datos procesados
│   ├── btc_raw.parquet
│   ├── btc_clean.parquet
│   ├── btc_with_technical_features.parquet
│   ├── btc_with_all_features.parquet
│   └── btc_with_target.parquet
│
├── features/                            # Datasets ML
│   ├── X_train.parquet
│   ├── X_test.parquet
│   ├── y_train.parquet
│   ├── y_test.parquet
│   ├── X_train_scaled.parquet
│   ├── X_test_scaled.parquet
│   ├── prices_train.parquet
│   ├── prices_test.parquet
│   └── feature_names.txt
│
├── models/                              # Modelos entrenados
│   ├── scaler.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── rf_feature_importance.csv
│   ├── gb_feature_importance.csv
│   └── test_results.json
│
├── predictions/                         # Predicciones diarias
│   ├── btc_recent.parquet
│   ├── btc_with_features.parquet
│   ├── prediccion_YYYYMMDD_HHMMSS.json
│   ├── historial_predicciones.jsonl
│   ├── notificacion_YYYYMMDD.txt
│   └── reporte_diario_YYYYMMDD.html
│
└── reports/                             # Reportes
    ├── plots/
    │   ├── metricas_comparacion.png
    │   ├── confusion_matrices.png
    │   └── feature_importance.png
    ├── feature_engineering_report_YYYYMMDD.html
    └── training_report_YYYYMMDD.html
```

---

## 🔧 Configuración de los DAGs

### DAG 1: bitcoin_data_pipeline

**Propósito:** Descargar datos históricos de Bitcoin

**Schedule:** Manual (ejecutar una vez al inicio)

**Tasks:**
1. Crear directorios
2. Descargar datos históricos (2 años)
3. Validar datos históricos
4. Descargar precio actual
5. Validar precio actual
6. Generar reporte

**Trigger siguiente:** `bitcoin_feature_engineering`

### DAG 2: bitcoin_feature_engineering

**Propósito:** Crear features para ML

**Schedule:** Triggered por DAG 1

**Tasks:**
1. Cargar datos históricos
2. Limpiar datos
3. Crear features técnicas (190+ features)
4. Crear features temporales
5. Crear variable objetivo
6. Preparar datasets (train/test split)
7. Generar reporte

**Features creadas:**
- 📈 Medias móviles (SMA, EMA, WMA)
- 💪 Momentum (RSI, ROC, CCI, Stochastic)
- 📊 MACD (múltiples configuraciones)
- 📏 Bandas (Bollinger, Keltner, Donchian)
- 🌊 Volatilidad (ATR, Parkinson)
- 📦 Volumen (OBV, MFI, VWAP)
- 🕐 Lag features
- 📊 Rolling statistics
- 🔢 Ratios y Fibonacci
- 📅 Features temporales

**Trigger siguiente:** `bitcoin_model_training`

### DAG 3: bitcoin_model_training

**Propósito:** Entrenar y evaluar modelos de ML

**Schedule:** Triggered por DAG 2

**Tasks:**
1. Cargar datasets
2. Normalizar features (StandardScaler)
3. Entrenar Logistic Regression
4. Entrenar Random Forest
5. Entrenar Gradient Boosting
6. Evaluar todos en test set
7. Generar gráficos comparativos
8. Generar reporte final

**Modelos entrenados:**
- Logistic Regression (baseline)
- Random Forest (mejor modelo ~58% accuracy)
- Gradient Boosting

**Outputs:**
- Modelos guardados (.pkl)
- Feature importance (CSV)
- Métricas de evaluación (JSON)
- Gráficos comparativos (PNG)
- Reporte HTML completo

### DAG 4: bitcoin_daily_prediction

**Propósito:** Predicción diaria automatizada

**Schedule:** `0 9 * * *` (todos los días a las 9 AM)

**Tasks:**
1. Descargar datos recientes
2. Crear features (mismas que en entrenamiento)
3. Hacer predicción con modelo entrenado
4. Enviar notificación
5. Generar reporte diario

**Outputs:**
- Predicción JSON
- Historial acumulado (JSONL)
- Notificación (TXT)
- Reporte HTML diario

---

## 🌐 API REST

### Iniciar la API

```bash
# Opción 1: Desde el host
python api/api_prediccion.py

# Opción 2: Desde el contenedor
docker exec -it <container-id> python /path/to/api_prediccion.py
```

La API corre en: **http://localhost:5000**

### Endpoints Disponibles

#### `GET /`
Información de la API

```bash
curl http://localhost:5000/
```

#### `GET /health`
Health check

```bash
curl http://localhost:5000/health
```

#### `GET /predict/now`
Predicción en tiempo real (descarga datos frescos)

```bash
curl http://localhost:5000/predict/now
```

**Respuesta:**
```json
{
  "timestamp": "2025-02-21T10:30:45",
  "precio_actual": 45678.32,
  "prediccion": {
    "direccion": "SUBE",
    "valor": 1,
    "emoji": "📈"
  },
  "probabilidades": {
    "sube": 0.673,
    "baja": 0.327
  },
  "confianza": 0.673,
  "indicadores": {
    "rsi_14": 54.23,
    "volatility_7": 0.023,
    "sma_7": 44892.15
  }
}
```

#### `GET /latest`
Última predicción guardada

```bash
curl http://localhost:5000/latest
```

#### `GET /history?limit=10`
Historial de predicciones

```bash
curl http://localhost:5000/history?limit=20
```

#### `GET /stats`
Estadísticas del sistema

```bash
curl http://localhost:5000/stats
```

---

## 📊 Dashboard Interactivo

### Iniciar el Dashboard

```bash
streamlit run dashboard/dashboard_streamlit.py
```

El dashboard abre en: **http://localhost:8501**

### Características

- 🔮 **Última predicción** con confianza
- 💰 **Precio actual** de Bitcoin
- 📈 **Gráfico interactivo** (Plotly) de historial
- 📊 **Distribución** de predicciones (pie chart)
- 🎯 **Estadísticas** generales del sistema
- 🤖 **Métricas del modelo** entrenado
- 📋 **Tabla completa** de historial
- 🔄 **Actualización** en tiempo real

---

## 📈 Métricas y Resultados

### Modelo Ganador: ???

```
Accuracy:   ?%
Precision:  ?%
Recall:     ?%
F1-Score:   ?
ROC-AUC:    ?
```

---

## 🔄 Flujo de Trabajo Diario

### Día 1 (Setup Inicial)

```
09:00 - Ejecutas bitcoin_data_pipeline manualmente
09:05 - Se dispara bitcoin_feature_engineering automáticamente
09:15 - Se dispara bitcoin_model_training automáticamente
09:35 - Sistema listo, modelo entrenado
```

### Día 2 en adelante (Automático)

```
09:00 - bitcoin_daily_prediction se ejecuta SOLO
09:01 - Descarga datos frescos
09:02 - Crea features
09:03 - Hace predicción: "SUBE 📈 67.3%"
09:04 - Guarda en historial
09:05 - Genera reporte HTML
09:06 - Listo, predicción disponible en API/Dashboard
```

---

## 🛠️ Scripts Útiles

### hacer_predicciones.py

Script para probar el modelo sin Airflow

```bash
python scripts/hacer_predicciones.py
```

**Funciones:**
- Predicción individual
- Últimos N días
- Simulación de trading
- Comparación vs buy & hold

### explorar_features.py

Explora los datasets procesados

```bash
python scripts/explorar_features.py
```

**Muestra:**
- Todas las features creadas
- Estadísticas de cada feature
- Correlaciones con el target
- Verificación de calidad

### diagnostico.py

Diagnostica problemas del sistema

```bash
docker exec -it <container> python /tmp/diagnostico.py
```

**Verifica:**
- Versiones de paquetes
- Imports funcionando
- Directorios creados
- Permisos correctos
- DAG válido

---

## 🐛 Troubleshooting

### Problema: DAGs no aparecen en Airflow UI

**Solución:**
```bash
# Verificar que están en la carpeta correcta
ls ~/airflow/dags/

# Reiniciar scheduler
docker-compose restart airflow-scheduler

# Ver logs del scheduler
docker logs <scheduler-container-id>
```

### Problema: Error "ModuleNotFoundError: No module named 'yfinance'"

**Solución:**
```bash
# Instalar en el contenedor correcto (worker)
docker exec -it <airflow-worker> pip install yfinance --break-system-packages
```

### Problema: DAG falla en "descargar_historicos"

**Solución:**
```bash
# Verificar conexión a internet
docker exec -it <container> ping -c 3 google.com

# Verificar que yfinance funciona
docker exec -it <container> python -c "import yfinance as yf; print(yf.download('BTC-USD', period='5d'))"
```

### Problema: Features no se crean correctamente

**Solución:**
```bash
# Verificar que numpy está instalado
docker exec -it <container> pip install numpy --break-system-packages

# Ver logs de la task
# En Airflow UI → DAG → Task → Log
```

### Problema: Modelo no se carga en predicción diaria

**Solución:**
```bash
# Verificar que existen los archivos del modelo
docker exec -it <container> ls /opt/airflow/data/models/

# Deberías ver:
# - random_forest.pkl
# - scaler.pkl
# - feature_names.txt

# Si no existen, ejecuta primero bitcoin_model_training
```

### Problema: API no inicia

**Solución:**
```bash
# Verificar Flask instalado
pip install flask flask-cors

# Ver errores
python api/api_prediccion.py
```

### Problema: Dashboard de Streamlit no abre

**Solución:**
```bash
# Verificar Streamlit instalado
pip install streamlit plotly

# Ejecutar con verbose
streamlit run dashboard/dashboard_streamlit.py --logger.level=debug
```

---

## 📚 Recursos y Referencias

### Documentación Oficial

- [Apache Airflow](https://airflow.apache.org/docs/)
- [Scikit-learn](https://scikit-learn.org/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Flask](https://flask.palletsprojects.com/)
- [Streamlit](https://docs.streamlit.io/)

### Análisis Técnico

- [Investopedia - Technical Indicators](https://www.investopedia.com/technical-analysis-4689657)
- [TA-Lib](https://ta-lib.org/)
- [TradingView](https://www.tradingview.com/scripts/)

### Machine Learning

- [Feature Engineering for ML](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Hands-On ML with Scikit-Learn](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

---

## 🔒 Consideraciones de Seguridad

### Para Producción Real

Este código es para **desarrollo y aprendizaje**. Para producción:

1. **Autenticación en API**
   ```python
   from flask_httpauth import HTTPBasicAuth
   auth = HTTPBasicAuth()
   ```

2. **Rate Limiting**
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app)
   ```

3. **HTTPS**
   - Usar certificados SSL
   - No exponer en HTTP

4. **Variables de entorno**
   ```python
   import os
   API_KEY = os.environ.get('API_KEY')
   ```

5. **Base de datos**
   - PostgreSQL en vez de JSON
   - Backups automáticos

6. **Monitoring**
   - Prometheus + Grafana
   - Alertas en Slack/PagerDuty

7. **CI/CD**
   - GitHub Actions
   - Tests automatizados
   - Deploy automático

---

## ⚠️ Disclaimer

**IMPORTANTE:** Este sistema es **EDUCATIVO**.

- ❌ **NO** es consejo financiero
- ❌ **NO** garantiza ganancias
- ❌ Las criptomonedas son **EXTREMADAMENTE** volátiles
- ❌ Puedes **PERDER TODO** tu dinero

**Si decides usar para trading real:**
- ✅ Empieza con cantidades **PEQUEÑAS**
- ✅ Usa **stop losses**
- ✅ **NUNCA** inviertas lo que no puedes perder
- ✅ Haz tu propia investigación (DYOR)
- ✅ Consulta un asesor financiero

**Los resultados pasados NO garantizan resultados futuros.**

---

## 🚀 Próximos Pasos

### Mejoras Sugeridas

1. **Re-entrenamiento automático**
   - DAG semanal que re-entrena con datos frescos
   - A/B testing de modelos

2. **Más modelos**
   - LSTM para series temporales
   - XGBoost
   - Ensemble de modelos

3. **Más features**
   - Sentiment analysis de Twitter
   - Google Trends
   - On-chain metrics
   - Datos macro (inflación, tasas)

4. **Trading bot**
   - Integración con Binance API
   - Gestión de riesgo
   - Backtesting robusto

5. **Mejor infraestructura**
   - Kubernetes para escalar
   - PostgreSQL para datos
   - Redis para caché
   - Grafana para monitoring

---

## 👥 Contribuciones

Este es un proyecto educativo. Sugerencias y mejoras son bienvenidas.

### Cómo contribuir

1. Fork el repositorio
2. Crea una rama para tu feature
3. Haz commit de tus cambios
4. Push a la rama
5. Abre un Pull Request

---

## 📄 Licencia

MIT License - Libre para usar, modificar y distribuir.

---

## 🎓 Sobre la Serie

Esta serie de videos fue creada para enseñar:
- ✅ Data Engineering con Airflow
- ✅ Feature Engineering para ML
- ✅ Machine Learning con Scikit-learn
- ✅ Deployment de modelos en producción
- ✅ APIs REST con Flask
- ✅ Dashboards con Streamlit

**Un proyecto completo end-to-end de ML.**

---

## 📞 Contacto y Soporte

- 📺 YouTube: https://www.youtube.com/c/Errodringer
- 💻 GitHub: https://github.com/errodringer
- 📧 Email: errodringer@gmail.com

---

## 🎉 Agradecimientos

Gracias por seguir la serie completa. Espero que hayas aprendido tanto como yo al crear este contenido.

**¡Feliz coding y trading responsable!** 🚀📈

---

**Última actualización:** Febrero 2026
**Versión:** 1.0.0
**Python:** 3.12
**Airflow:** 2.9.3
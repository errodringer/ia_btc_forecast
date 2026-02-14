# 🚀 Pipeline de Datos de Bitcoin con Airflow

## 📺 Video Tutorial - Parte 2

Este es el código completo para el **segundo video** de la serie sobre predicción de Bitcoin con ML.

---

## 🎯 ¿Qué hace este DAG?

El DAG `bitcoin_data_pipeline` ejecuta un pipeline completo que:

1. **📥 Descarga datos históricos** - 2 años de precios de Bitcoin desde Yahoo Finance
2. **✅ Valida los datos históricos** - Chequea gaps, outliers, valores negativos
3. **💎 Descarga precio actual** - Precio en tiempo real desde CoinGecko API
4. **✅ Valida precio actual** - Verifica que el precio sea razonable
5. **📊 Genera reporte HTML** - Un dashboard visual con todas las estadísticas

---

## 🛠️ Setup Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Copiar el DAG a Airflow

```bash
# Si estás usando Docker (lo más común)
cp bitcoin_data_pipeline.py ~/airflow/dags/

# O si instalaste Airflow localmente
cp bitcoin_data_pipeline.py $AIRFLOW_HOME/dags/
```

### 3. Crear las carpetas de datos

```bash
mkdir -p /opt/airflow/data/{historical,current,reports}
```

**Nota:** Si usas Docker, estas carpetas se crearán automáticamente dentro del contenedor.

### 4. Activar el DAG

1. Abre Airflow UI: `http://localhost:8080`
2. Busca el DAG `bitcoin_data_pipeline`
3. Actívalo con el toggle
4. Dale click a "Trigger DAG" para ejecutarlo manualmente

---

## 📊 Estructura del Pipeline

```
crear_directorios
       ├─→ descargar_historicos → validar_historicos ─┐
       │                                                ├─→ generar_reporte
       └─→ descargar_precio_actual → validar_actual ──┘
```

---

## 🎬 Tips para el Video

### 🔥 Momentos "Wow" para mostrar:

1. **El Graph View en Airflow UI** - Muestra el flujo visual del DAG
2. **Los logs en tiempo real** - Abre los logs mientras corre y muestra los emojis 💰📊✅
3. **El reporte HTML generado** - Abre el archivo HTML y muestra las estadísticas
4. **Simula un fallo** - Modifica temporalmente el código para que falle (ej: URL incorrecta) y muestra cómo Airflow lo reintenta
5. **El calendario de ejecuciones** - Muestra las ejecuciones programadas

### 📝 Puntos clave para explicar:

- **XCom**: Cómo las tasks se pasan datos entre sí
- **Retry logic**: Por qué configuramos `retries=2`
- **Schedule**: `'0 9 * * *'` corre a las 9 AM todos los días
- **Parquet vs CSV**: Por qué usamos parquet (más eficiente)
- **Validaciones**: Por qué es importante validar los datos antes de entrenar el modelo

### 🎨 Customizaciones para hacer en vivo:

```python
# Cambia el schedule para que corra cada 5 minutos (para el demo)
schedule_interval='*/5 * * * *',

# O cambia los emojis en los logs
logging.info("🎉 ¡Datos descargados con éxito!")
```

---

## 📁 Archivos Generados

Después de ejecutar el DAG, encontrarás:

```
/opt/airflow/data/
├── historical/
│   └── btc_historical_20240213.parquet  # Datos históricos
├── current/
│   └── btc_current_20240213_093045.json  # Precio actual
└── reports/
    └── reporte_20240213_093050.html      # Reporte visual
```

---

## 🐛 Troubleshooting

### Error: "No module named 'yfinance'"
```bash
pip install yfinance
```

### Error: "Permission denied" al crear carpetas
```bash
# Cambia los permisos
sudo chmod -R 777 /opt/airflow/data
```

### El DAG no aparece en Airflow UI
```bash
# Verifica que el archivo esté en la carpeta correcta
ls ~/airflow/dags/

# Reinicia Airflow
docker-compose restart
# O si es local:
airflow scheduler restart
```

---

## 🎓 Próximos Pasos (Video 3)

En el próximo video usaremos estos datos para:
- Explorar los datos con pandas
- Crear features para el modelo de ML
- Entrenar un modelo simple de predicción
- ¡Comparar predicciones vs realidad!

---

## 📚 Recursos Adicionales

- [Documentación de Airflow](https://airflow.apache.org/docs/)
- [Yahoo Finance API](https://github.com/ranaroussi/yfinance)
- [CoinGecko API](https://www.coingecko.com/en/api)

---

## 💡 Ideas para Mejorar (para futuros videos)

- [ ] Añadir alertas por email cuando Bitcoin sube/baja mucho
- [ ] Guardar datos en una base de datos PostgreSQL
- [ ] Añadir más fuentes de datos (Twitter sentiment, Google Trends)
- [ ] Crear un dashboard en Grafana
- [ ] Añadir tests unitarios para las funciones

---

¡Happy coding! 🎉

Si este código te sirvió, no olvides:
- 👍 Darle like al video
- 🔔 Suscribirte al canal
- 💬 Dejar un comentario con dudas o sugerencias

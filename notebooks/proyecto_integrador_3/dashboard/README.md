# RiskMap PA — Sistema Predictivo de Accidentes de Tránsito

Dashboard comercial en Streamlit construido sobre datos calibrados INEC 2023-2024 + FEDPA. 
Utiliza modelo predictivo preentrenado (Random Forest + Poisson) exportado desde `proyecto_final.ipynb`.

## Archivos requeridos

```
app.py
requirements.txt
accident_prediction_system.joblib   ← modelo preentrenado (joblib)
panama_synthetic_accidents.csv      ← dataset de calibración
```

Todos los archivos deben estar en el mismo directorio.

## Estructura del modelo cargado

El objeto `sistema` (joblib) contiene:

- **`predict_severity(input_df)`** → DataFrame con columnas:
  - `prob_Menor`, `prob_Intermedio`, `prob_Mayor` (probabilidades calibradas INEC)
  - Severidad predicha según clasificación MUTCD
  
- **`poisson_model`** → Predictor de frecuencia (E[accidentes/zona-hora])

- Componentes internos:
  - Pipeline de preprocessing (escalado numérico + encoding categórico)
  - Clasificador Random Forest (120 estimadores)
  - Factores de calibración INEC por corregimiento

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre en: http://localhost:8501

## Deploy en Streamlit Cloud (gratis)

1. Sube los 4 archivos a un repo GitHub (puede ser privado)
2. Ve a https://share.streamlit.io → New app
3. Selecciona el repo, branch `main`, archivo `app.py`
4. Deploy automático

## Deploy en Google Cloud Run

```dockerfile
# Dockerfile mínimo
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

```bash
gcloud run deploy riskmap-pa \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## Flujo de predicción

1. **Carga del modelo** → `@st.cache_resource` decora `load_model()` para caché persistente
2. **Entrada de usuario** → Parámetros clima, infraestructura, ubicación en sidebar
3. **Predicción de severidad** → `sistema.predict_severity(input_row)`
4. **Calibración INEC** → Ajuste de probabilidades según peso provincial
5. **Visualización** → KPIs, gauge chart, mapa interactivo con Folium

## Features del dashboard

### Tab 1 — Predictor
- Predicción en tiempo real de severidad (MENOR/INTERMEDIO/MAYOR)
- Probabilidades calibradas con pesos INEC
- Índice de prima técnica relativa por corregimiento
- Mapa interactivo con puntos de accidentes históricos

### Tab 2 — Mapa de Riesgo
- Mapa de calor a nivel de corregimiento
- Circulos proporcionales a accidentalidad INEC 2024
- Variación YoY por zona
- Popup con estadísticas de severidad

### Tab 3 — Análisis
- Distribución de accidentes por hora, día, mes
- Análisis de factores de riesgo (clima, infraestructura)
- Series temporales de YoY growth

### Tab 4 — Actuarial
- Tabla de primas técnicas por corregimiento
- Comparativa INEC 2023 vs 2024
- Factores de ajuste por riesgo

## Fuentes de calibración

- **INEC 2024**: 23,235 accidentes / 25 corregimientos / Distrito de Panamá
- **INEC 2023**: 21,801 accidentes (YoY +6.6%)
- **FEDPA**: 517 reclamos reales deidentificados (broker panameño)
- **Base del modelo**: Florida Accidents Dataset (US_Accidents_FL.csv)

## Personalización

### Cambiar dataset de calibración
Reemplazar `panama_synthetic_accidents.csv` en `load_data()` (línea 177)

### Ajustar pesos de calibración
Los factores INEC se cargan desde el CSV. Actualizar columnas `INEC_weight`, `INEC_2024`, `INEC_2023`

### Modificar factores de conversión
La calibración Platt en línea ~297 puede ajustarse:
```python
cal_fac = zone_row["INEC_weight"] / mean_w  # multiplicador por zona
```

## Solución de problemas

**Error: "accident_prediction_system.joblib no encontrado"**
- Verifica que el archivo esté en el mismo directorio que `app.py`
- Confirma el nombre exacto del archivo

**Modelo predice siempre la misma clase**
- Verifica que el joblib contiene un modelo entrenado correctamente
- Comprueba que `input_row` contiene todas las columnas requeridas

**Dashboard lento**
- `@st.cache_resource` y `@st.cache_data` ya están optimizados
- Para >100k registros, considera subsampling en `load_data()`


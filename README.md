# Modelo Predictivo de Ocurrencia y Nivel de Severidad de Accidentes Automovilísticos

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat-square&logo=pandas&logoColor=white)

**Universidad Tecnológica de Panamá**  
Maestría en Analítica de Datos — Proyectos Integradores I, II y III · 2026

</div>

---

## Descripción del Proyecto

Este repositorio contiene el desarrollo de un **Modelo Predictivo de Ocurrencia y Nivel de Severidad de Accidentes Automovilísticos** utilizando el dataset **US Accidents**, filtrado para el **estado de Florida** como proxy de las condiciones de tráfico en Panamá. El proyecto abarca tres fases académicas secuenciales:

| Fase | Materia | Enfoque |
|------|---------|---------|
| Fase I | Proyecto Integrador I | Exploración, limpieza y análisis descriptivo de datos |
| Fase II | Proyecto Integrador II | Modelado predictivo avanzado (ocurrencia + severidad) |
| Fase III | Proyecto Integrador III | Dashboard comercial calibrado con datos INEC Panamá |

### Objetivos

1. **Estimar la probabilidad de ocurrencia** de accidentes por zona geográfica y condiciones ambientales.
2. **Predecir el nivel de severidad** de accidentes en tres categorías MUTCD: `Menor`, `Intermedio` y `Mayor`.
3. **Desplegar un dashboard interactivo** (RiskMap PA) con calibración actuarial para el mercado panameño.

### Por qué Florida

Florida fue seleccionado como estado de referencia por su **similitud estructural con Panamá**:
- Clima subtropical con lluvias intensas
- Infraestructura mixta urbana/rural
- Geografía costera plana
- Patrones de tráfico y densidad vehicular comparables

---

## Estructura del Repositorio

```diff
 final_project/
 ├── .gitignore
 ├── README.md
 └── notebooks/
     ├── data/                                           # Datos de entrada (origen: Kaggle)
     │   ├── US_Accidents_FL.csv                         # Dataset filtrado (Florida, ~270K registros)
     │   └── US_Accidents_encoded.csv                    # Dataset procesado y codificado
     ├── output/                                         # Artefactos generados por el pipeline
     │   ├── charts/                                     # Gráficos exportados por los notebooks
     │   └── data/                                       # CSVs y JSON generados
     │       ├── panama_synthetic_accidents.csv          # Dataset base calibrado con INEC 2023-2024
     │       ├── panama_synthetic_accidents_weather.csv  # Dataset enriquecido con ERA5 (Open-Meteo)
     │       ├── panama_synthetic_accidents_weather_v4.csv  # Versión v4 del dataset meteorológico
     │       ├── panama_severity_dist.json               # Distribución de severidad INEC (priors actuariales)
     │       ├── inec_hour_dow_joint.csv                 # Distribución conjunta hora-día (INEC 2024)
     │       └── inec_road_dist.json                     # Distribución de riesgo por tipo de vía (INEC)
     ├── proyecto_integrador_1/                          # PROYECTO INTEGRADOR I
     │   ├── 01.download_dataset.ipynb                   # Descarga del dataset vía KaggleHub
     │   ├── 02.EDA.ipynb                                # Análisis exploratorio inicial (Pandas)
     │   ├── 02.DASK_EDA.ipynb                           # EDA con Dask (escalabilidad big data)
     │   └── 03.EDA_FL.ipynb                             # EDA específico para Florida
     ├── proyecto_integrador_2/                          # PROYECTO INTEGRADOR II
     │   ├── 04.EDA_FL_v2.ipynb                          # EDA avanzado + modelos predictivos
     │   └── 04.EDA_FL_v2.1.ipynb                        # Versión refinada con correcciones y mejoras
+    └── proyecto_integrador_3/                          # PROYECTO INTEGRADOR III (ACTUAL)
         ├── proyecto_final.ipynb                        # Entrenamiento del modelo final (Colab)
         ├── validacion_capas_sintetico.ipynb            # Validación del dataset sintético por capas
         ├── utils/
         │   ├── weather_enrichment.py                   # Enriquecimiento de datos con Open-Meteo
         │   └── weather_checkpoint.csv                  # Checkpoint de progreso del enriquecimiento
         └── dashboard/                                  # Dashboard comercial RiskMap PA
             ├── app.py                                  # Aplicación Streamlit principal
             ├── model.py                                # Definición standalone de AccidentPredictionSystem
             ├── requirements.txt                        # Dependencias del dashboard
             ├── smoke_test.py                           # Tests de humo para validar el pipeline
             ├── weather_checkpoint.csv                  # Caché de condiciones meteorológicas
             └── accident_prediction_system.joblib       # Modelo preentrenado (~6.2 GB, joblib)
```

---

## Proyecto Integrador I — Fundamentos

### Notebooks

#### [`01.download_dataset.ipynb`](notebooks/proyecto_integrador_1/01.download_dataset.ipynb)
Descarga automática del dataset **US Accidents** desde Kaggle usando la API `kagglehub`.
- **Output**: Dataset crudo almacenado localmente (~3M registros, 46+ columnas).

#### [`02.EDA.ipynb`](notebooks/proyecto_integrador_1/02.EDA.ipynb) · [`02.DASK_EDA.ipynb`](notebooks/proyecto_integrador_1/02.DASK_EDA.ipynb)
Análisis exploratorio del dataset completo.
- **Pandas**: estadísticas descriptivas, distribuciones, calidad de datos.
- **Dask**: misma exploración pero optimizada para grandes volúmenes de datos.
- **Output**: `US_Accidents_FL.csv` — dataset filtrado para Florida (~270K registros).

#### [`03.EDA_FL.ipynb`](notebooks/proyecto_integrador_1/03.EDA_FL.ipynb)
Análisis profundo sobre el subconjunto de Florida.
- Filtrado geográfico y limpieza de variables relevantes.
- Visualización de patrones de accidentalidad por condado, hora y clima.

---

## Proyecto Integrador II — Modelado Predictivo

### Notebooks

#### [`04.EDA_FL_v2.ipynb`](notebooks/proyecto_integrador_2/04.EDA_FL_v2.ipynb) — Notebook Principal
El notebook central del proyecto. Integra análisis avanzado y modelado en un flujo narrativo de *data storytelling*.

**Contenido:**
- Análisis granular por County y City
- Patrones temporales: hora, día, mes, estación del año
- Impacto de condiciones climáticas: temperatura, humedad, visibilidad
- Análisis de infraestructura vial: semáforos, intersecciones, cruces ferroviarios
- **Etapa 1 — Ocurrencia**: Análisis descriptivo de distribución de accidentes por zona geográfica (nivel Bajo / Moderado / Alto / Crítico por conteos)
- **Etapa 2 — Severidad**: Clasificación `Low / Moderate / High` con Random Forest, Extra Trees y XGBoost (validación cruzada 5-fold + comparación de modelos)
- Visualizaciones interactivas con Plotly

#### [`04.EDA_FL_v2.1.ipynb`](notebooks/proyecto_integrador_2/04.EDA_FL_v2.1.ipynb) — Análisis Complementario
Análisis adicional elaborado a partir de consultas del profesor para profundizar en el impacto de features específicos sobre la severidad y actualizar gráficas ya entregadas en el Proyecto Integrador II.

### Variables Clave

| Categoría | Variables |
|-----------|-----------|
| Geográficas | County, City, Start_Lat, Start_Lng |
| Temporales | Hour, DayOfWeek, Month, wet_season |
| Climatológicas | Weather_Condition, Temperature(F), Humidity(%), Visibility(mi), Precipitation(in) |
| Infraestructura | Traffic_Signal, Junction, Crossing, Roundabout, Amenity |
| Variable Objetivo | Severity → `Low` / `Moderate` / `High` (reagrupación de Severity 1–4) |

---

## Proyecto Integrador III — Dashboard Comercial (RiskMap PA)

### Descripción

**RiskMap PA** es un dashboard comercial en Streamlit que presenta el modelo predictivo calibrado con datos reales panameños (INEC 2023-2024 + Aseguradora Panameña). Está orientado a una audiencia de seguros/actuarial y permite evaluar el riesgo de accidentalidad por corregimiento del Distrito de Panamá en tiempo real.

### Archivos

#### [`proyecto_final.ipynb`](notebooks/proyecto_integrador_3/proyecto_final.ipynb)
Notebook de entrenamiento del modelo final (diseñado para ejecutarse en Google Colab).
- Entrena el pipeline completo: preprocessing + Random Forest calibrado + modelo Poisson
- Exporta `accident_prediction_system.joblib` para uso en el dashboard

**Métricas del modelo entrenado (test set, n=617,735):**

| Métrica | Valor |
|---------|-------|
| Accuracy | 60.9% |
| ROC AUC (weighted) | 0.794 |
| AUC clase Mayor | 0.836 |
| Gini | 0.671 |

#### [`validacion_capas_sintetico.ipynb`](notebooks/proyecto_integrador_3/validacion_capas_sintetico.ipynb)
Notebook de validación del dataset sintético de Panamá. Verifica la coherencia estadística del proceso de calibración por capas (INEC, ERA5, MUTCD) antes de alimentar el modelo y el dashboard.

#### [`utils/weather_enrichment.py`](notebooks/proyecto_integrador_3/utils/weather_enrichment.py)
Módulo de enriquecimiento meteorológico que consume la API **Open-Meteo** para asociar condiciones climáticas históricas (temperatura, precipitación, visibilidad) a cada registro del dataset sintético de Panamá.
- Lee desde `notebooks/output/data/panama_synthetic_accidents.csv`
- Implementa caché local (`utils/.openmeteo_cache`) y checkpoint (`utils/weather_checkpoint.csv`) para minimizar llamadas a la API
- Genera `notebooks/output/data/panama_synthetic_accidents_weather.csv` como output del pipeline

#### [`dashboard/app.py`](notebooks/proyecto_integrador_3/dashboard/app.py)
Aplicación Streamlit principal con cinco pestañas:

| Pestaña | Contenido |
|---------|-----------|
| EDA · Florida | Análisis exploratorio del dataset de entrenamiento (FL): severidad, patrones temporales, clima e infraestructura que motivaron las features del modelo |
| Mapa de Riesgo | Mapa de calor a nivel de corregimiento con datos INEC 2024 |
| Perfil de Siniestralidad | Distribución horaria, estacional, YoY INEC 2023 vs 2024 |
| Predictor | Predicción en tiempo real con condiciones meteorológicas automáticas (forecast Open-Meteo en tiempo real o climatología ERA5 como fallback); muestra severidad MUTCD, probabilidades calibradas INEC e índice de prima técnica relativa |
| Actuarial | Tabla y scatter de índice de prima técnica relativa por corregimiento (ver detalle abajo) |

#### Pestaña Actuarial — detalle de cálculos

La pestaña presenta dos vistas complementarias para la tarificación por corregimiento:

**Tabla de primas técnicas**  
Cada fila es un corregimiento ordenado de mayor a menor índice de prima. Las columnas son:

| Columna | Descripción |
|---------|-------------|
| INEC 2024 | Conteo real de accidentes según el Instituto Nacional de Estadística y Censo |
| YoY | Crecimiento interanual 2023→2024; un valor positivo encarece la prima |
| P(Mayor) | Probabilidad de que un accidente sea clasificado como Mayor (severidad alta) según el modelo Random Forest |
| Peso INEC | Factor de escala = INEC_2024 / media_distrital; normaliza la exposición de cada corregimiento respecto al promedio del Distrito |
| Índice Prima | Índice relativo de prima técnica (ver fórmula abajo) |

**Fórmula del índice de prima relativa:**

```
índice_prima(zona) = ( INEC_weight(zona) × P(Mayor|zona) × (1 + YoY(zona)) )
                     ────────────────────────────────────────────────────────
                          min( mismo producto sobre todas las zonas )
```

El índice es adimensional y relativo: la zona de menor riesgo compuesto vale `1.00×`; las demás se expresan como múltiplo de esa base. Un índice de `3.5×` significa que esa zona tiene una exposición actuarial 3.5 veces mayor que la de menor riesgo.

**Ejemplo de cálculo manual (3 zonas ilustrativas):**

| Corregimiento | INEC 2024 | Peso INEC | P(Mayor) | YoY | Producto crudo |
|---|---|---|---|---|---|
| Bella Vista | 2,881 | 3.10 | 18.5% | +8.2% | 3.10 × 0.185 × 1.082 = **0.6200** |
| Betania | 1,240 | 1.33 | 14.1% | +5.0% | 1.33 × 0.141 × 1.050 = **0.1970** |
| Chilibre | 180 | 0.19 | 11.8% | +1.1% | 0.19 × 0.118 × 1.011 = **0.0227** |

En el ejemplo, el mínimo del producto crudo entre todas las zonas es el de Chilibre: `0.0227`.

```
índice_prima(Bella Vista) = 0.6200 / 0.0227 = 27.3×
índice_prima(Betania)     = 0.1970 / 0.0227 =  8.7×
índice_prima(Chilibre)    = 0.0227 / 0.0227 =  1.0×  ← base
```

Interpretación: asegurar un vehículo en Bella Vista debería costar ~27 veces más que en Chilibre, y ~3 veces más que en Betania, considerando únicamente frecuencia observada, severidad modelada y tendencia de crecimiento. Los valores anteriores son ilustrativos; los índices reales se calculan sobre los 25 corregimientos del Distrito y aparecen en la tabla del dashboard.

**Scatter de índice de prima**  
Eje X = accidentes INEC 2024 (frecuencia observada), eje Y = P(Mayor) del modelo (severidad condicional), tamaño del punto = índice de prima. Permite identificar visualmente tres perfiles de riesgo:
- **Alta frecuencia + alta severidad** → prima más cara (punto grande, arriba a la derecha)
- **Alta frecuencia + baja severidad** → volumen sin mortalidad (punto chico, abajo a la derecha)
- **Baja frecuencia + alta severidad** → riesgo catastrófico puntual (punto grande, arriba a la izquierda)

**Conexión con el modelo de dos etapas:**  
La prima técnica completa requeriría `E[Frecuencia] × P(Mayor) × E[Costo siniestro] × (1 + loading)`. El índice usa `E[Frecuencia]` aproximado por el peso INEC (datos reales), `P(Mayor)` del Random Forest calibrado, y el factor YoY como proxy del loading de tendencia. El término `E[Costo siniestro]` está pendiente de datos reales de liquidación (fuente sugerida: Aseguradora Panameña).

---

#### [`dashboard/model.py`](notebooks/proyecto_integrador_3/dashboard/model.py)
Definición standalone de `AccidentPredictionSystem` — necesaria para deserializar el modelo `.joblib` fuera del entorno de entrenamiento (Colab).

#### [`dashboard/smoke_test.py`](notebooks/proyecto_integrador_3/dashboard/smoke_test.py)
Tests de humo para validar que el pipeline de carga del modelo y las predicciones de muestra producen salidas coherentes antes del despliegue.

### Fuentes de Calibración (Fase III)

| Fuente | Detalle |
|--------|---------|
| INEC 2024 | 23,235 accidentes / 25 corregimientos / Distrito de Panamá |
| INEC 2023 | 21,801 accidentes (YoY +6.6%) |
| INEC Vías | Distribución de riesgo por vía (micro-segmentación) y tipo de carretera → `inec_road_dist.json` |
| Aseguradora Panameña | 517 reclamos reales deidentificados (broker panameño) |
| Base del modelo | Florida Accidents Dataset (US_Accidents_FL.csv) |
| Open-Meteo API | Datos meteorológicos históricos (enriquecimiento offline) y pronóstico en tiempo real (predictor del dashboard) |

### Micro-segmentación de Riesgo

El modelo integra una capa de micro-segmentación tarifaria a nivel de vía y tipo de carretera (`Road_Type`), calculada a partir de estadísticas del INEC:
- Permite pasar de un análisis macro por corregimiento a un riesgo específico por ruta.
- Cuantifica y visualiza vías críticas con mayor `P(Mayor)` (`highway`: 0.85%, `street`: 0.79%, `avenue`: 0.41%).
- Introduce factores de recargo actuariales para el cálculo final de la prima técnica automotriz.

### Ejecutar el Dashboard

```bash
cd notebooks/proyecto_integrador_3/dashboard
pip install -r requirements.txt
streamlit run app.py
```

Abre en: http://localhost:8501

> **Nota**: El archivo `accident_prediction_system.joblib` (~6.2 GB) **está incluido** en el repositorio pero **no se sincroniza con GitHub** por su tamaño (supera el límite de 100 MB por archivo de Git). Si no está disponible localmente, ejecútalo desde `proyecto_final.ipynb` en Colab o solicítalo al equipo.

### Consideraciones para despliegue en la nube

El modelo `accident_prediction_system.joblib` pesa ~6.2 GB, lo que impone restricciones importantes:

| Plataforma | Límite RAM | Viable | Motivo |
|------------|-----------|--------|--------|
| Streamlit Community Cloud | ~1 GB | No | El modelo no cabe en memoria |
| GitHub | 100 MB por archivo | No | El `.joblib` supera el límite |
| Google Colab | 12–51 GB | Sí (demo) | Requiere ejecutar manualmente |
| VM / servidor propio | Sin límite fijo | Sí | Opción de producción |

**Opción recomendada — ejecución local:**

```bash
# El dashboard corre localmente sin restricciones de tamaño
cd notebooks/proyecto_integrador_3/dashboard
pip install -r requirements.txt
streamlit run app.py
```

**Opción alternativa — cargar el modelo desde Google Drive:**

Si se requiere compartir el dashboard sin ejecutarlo localmente, el modelo puede alojarse en Google Drive y descargarse al iniciar la app:

```bash
pip install gdown
```

```python
# Añadir al inicio de app.py antes de joblib.load()
import gdown, os
MODEL_PATH = "accident_prediction_system.joblib"
GDRIVE_ID  = "<file-id-de-google-drive>"
if not os.path.exists(MODEL_PATH):
    gdown.download(id=GDRIVE_ID, output=MODEL_PATH, quiet=False)
```

> **Nota**: Para una demo académica, la ejecución local es suficiente. Un despliegue cloud real requeriría reducir el modelo (quantización, menos estimadores) o contratar una VM con RAM >= 16 GB.

---

## Stack Tecnológico

| Librería | Versión | Uso |
|----------|---------|-----|
| `pandas` | >= 2.0 | Manipulación y limpieza de datos |
| `numpy` | >= 1.24 | Operaciones numéricas |
| `matplotlib` | >= 3.6 | Visualización estática |
| `seaborn` | >= 0.12 | Visualización estadística |
| `plotly` | >= 5.18 | Visualizaciones interactivas |
| `scikit-learn` | >= 1.3 | Modelado predictivo y evaluación |
| `dask` | >= 2023.x | Procesamiento escalable (big data) |
| `kagglehub` | latest | Descarga de datasets desde Kaggle |
| `streamlit` | >= 1.32 | Dashboard comercial (Fase III) |
| `folium` | >= 0.15 | Mapas interactivos (Fase III) |
| `streamlit-folium` | >= 0.18 | Integración Folium en Streamlit |
| `openmeteo-requests` | latest | Cliente API Open-Meteo (enriquecimiento meteorológico) |
| `joblib` | built-in sklearn | Serialización del modelo |

---

## Cómo Ejecutar el Proyecto

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd final_project
```

### 2. Instalar dependencias

**Fases I y II (notebooks):**
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn dask kagglehub jupyter
```

**Fase III (dashboard):**
```bash
pip install -r notebooks/proyecto_integrador_3/dashboard/requirements.txt
```

> Se recomienda usar un entorno virtual (`venv` o `conda`).

### 3. Configurar Kaggle API

Para descargar el dataset automáticamente, asegúrate de tener configurado el archivo `~/.kaggle/kaggle.json` con tus credenciales de Kaggle.

### 4. Flujo de ejecución

```
Proyecto Integrador I:
  01.download_dataset.ipynb  →  02.EDA.ipynb  →  03.EDA_FL.ipynb
                                                   └─ output: notebooks/data/US_Accidents_FL.csv

Proyecto Integrador II:
  04.EDA_FL_v2.ipynb  (requiere notebooks/data/US_Accidents_FL.csv)

Proyecto Integrador III:
  proyecto_final.ipynb            →  genera accident_prediction_system.joblib (en dashboard/)
  utils/weather_enrichment.py     →  lee   notebooks/output/data/panama_synthetic_accidents.csv
                                     genera notebooks/output/data/panama_synthetic_accidents_weather.csv
  streamlit run dashboard/app.py  →  lee   notebooks/output/data/
```

> **Nota**: Los archivos CSV del dataset y el modelo `.joblib` **no están incluidos** en el repositorio por su tamaño. Ejecuta `01.download_dataset.ipynb` primero para generarlos.

---

## Dataset

| Atributo | Detalle |
|----------|---------|
| Nombre | [US Accidents (2016-2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) |
| Fuente | Kaggle — Sobhan Moosavi |
| Tamaño original | ~7.7 millones de registros |
| Subconjunto Florida | ~270K registros |
| Periodo | 2016 – 2023 |
| Columnas | 46+ features |

---

## Equipo

**Grupo 1** — Maestría en Analítica de Datos  
Universidad Tecnológica de Panamá · 2026

---

## Licencia

Este proyecto es de uso académico. El dataset US Accidents está sujeto a los [términos de uso de Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents).

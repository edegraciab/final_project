# Modelo Predictivo de Accidentes Automovilísticos

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

Este repositorio contiene el desarrollo de un **modelo predictivo de accidentes automovilísticos** utilizando el dataset **US Accidents**, filtrado para el **estado de Florida** como proxy de las condiciones de tráfico en Panamá. El proyecto abarca tres fases académicas secuenciales:

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

```
final_project/
├── .gitignore
├── README.md
└── notebooks/
    ├── data/                                       # Datos de entrada (origen: Kaggle)
    │   ├── US_Accidents_FL.csv                     # Dataset filtrado (Florida)
    │   └── US_Accidents_encoded.csv                # Dataset procesado y codificado
    ├── output/                                     # Artefactos generados por el pipeline
    │   ├── charts/                                 # Gráficos exportados por los notebooks
    │   └── data/                                   # CSVs y JSON generados
    │       ├── panama_synthetic_accidents.csv      # Dataset base calibrado con INEC 2023-2024
    │       ├── panama_synthetic_accidents_weather.csv  # Dataset enriquecido con ERA5 (Open-Meteo)
    │       ├── panama_severity_dist.json           # Distribución de severidad INEC (priors actuariales)
    │       └── inec_hour_dow_joint.csv             # Distribución conjunta hora-día (INEC 2024)
    ├── proyecto_integrador_1/                      # PROYECTO INTEGRADOR I
    │   ├── 01.download_dataset.ipynb               # Descarga del dataset vía KaggleHub
    │   ├── 02.EDA.ipynb                            # Análisis exploratorio inicial (Pandas)
    │   ├── 02.DASK_EDA.ipynb                       # EDA con Dask (escalabilidad big data)
    │   └── 03.EDA_FL.ipynb                         # EDA específico para Florida
    ├── proyecto_integrador_2/                      # PROYECTO INTEGRADOR II
    │   └── 04.EDA_FL_v2.ipynb                      # EDA avanzado + modelos predictivos
    └── proyecto_integrador_3/                      # PROYECTO INTEGRADOR III (ACTUAL)
        ├── proyecto_final.ipynb                    # Entrenamiento del modelo final (Colab)
        ├── utils/
        │   └── weather_enrichment.py               # Enriquecimiento de datos con Open-Meteo
        └── dashboard/                              # Dashboard comercial RiskMap PA
            ├── app.py                              # Aplicación Streamlit principal
            ├── model.py                            # Definición standalone de AccidentPredictionSystem
            ├── requirements.txt                    # Dependencias del dashboard
            ├── smoke_test.py                       # Tests de humo para validar el pipeline
            └── accident_prediction_system.joblib   # Modelo preentrenado (joblib)
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
- **Modelo de Ocurrencia**: Probabilidad de accidente por zona (Poisson)
- **Modelo de Severidad**: Clasificación `Menor / Intermedio / Mayor` (MUTCD)
- Visualizaciones interactivas con Plotly

### Variables Clave

| Categoría | Variables |
|-----------|-----------|
| Geográficas | County, City, Start_Lat, Start_Lng |
| Temporales | Hour, DayOfWeek, Month, wet_season |
| Climatológicas | Weather_Condition, Temperature(F), Humidity(%), Visibility(mi), Precipitation(in) |
| Infraestructura | Traffic_Signal, Junction, Crossing, Roundabout, Amenity |
| Variable Objetivo | Severity → `Menor` / `Intermedio` / `Mayor` (MUTCD) |

---

## Proyecto Integrador III — Dashboard Comercial (RiskMap PA)

### Descripción

**RiskMap PA** es un dashboard comercial en Streamlit que presenta el modelo predictivo calibrado con datos reales panameños (INEC 2023-2024 + FEDPA). Está orientado a una audiencia de seguros/actuarial y permite evaluar el riesgo de accidentalidad por corregimiento del Distrito de Panamá en tiempo real.

### Archivos

#### [`proyecto_final.ipynb`](notebooks/proyecto_integrador_3/proyecto_final.ipynb)
Notebook de entrenamiento del modelo final (diseñado para ejecutarse en Google Colab).
- Entrena el pipeline completo: preprocessing + Random Forest calibrado + modelo Poisson
- Exporta `accident_prediction_system.joblib` para uso en el dashboard

#### [`utils/weather_enrichment.py`](notebooks/proyecto_integrador_3/utils/weather_enrichment.py)
Módulo de enriquecimiento meteorológico que consume la API **Open-Meteo** para asociar condiciones climáticas históricas (temperatura, precipitación, visibilidad) a cada registro del dataset sintético de Panamá.
- Lee desde `notebooks/output/data/panama_synthetic_accidents.csv`
- Implementa caché local (`utils/.openmeteo_cache`) y checkpoint (`utils/weather_checkpoint.csv`) para minimizar llamadas a la API
- Genera `notebooks/output/data/panama_synthetic_accidents_weather.csv` como output del pipeline

#### [`dashboard/app.py`](notebooks/proyecto_integrador_3/dashboard/app.py)
Aplicación Streamlit principal con cuatro pestañas:

| Pestaña | Contenido |
|---------|-----------|
| Predictor | Predicción en tiempo real (severidad MUTCD + probabilidades calibradas INEC + índice de prima técnica relativa) |
| Mapa de Riesgo | Mapa de calor a nivel de corregimiento con datos INEC 2024 |
| Análisis | Distribución horaria, estacional, YoY INEC 2023 vs 2024, precipitación y visibilidad |
| Actuarial | Tabla de primas técnicas por corregimiento + scatter de índice de prima |

#### [`dashboard/model.py`](notebooks/proyecto_integrador_3/dashboard/model.py)
Definición standalone de `AccidentPredictionSystem` — necesaria para deserializar el modelo `.joblib` fuera del entorno de entrenamiento (Colab).

#### [`dashboard/smoke_test.py`](notebooks/proyecto_integrador_3/dashboard/smoke_test.py)
Tests de humo para validar que el pipeline de carga del modelo y las predicciones de muestra producen salidas coherentes antes del despliegue.

### Fuentes de Calibración (Fase III)

| Fuente | Detalle |
|--------|---------|
| INEC 2024 | 23,235 accidentes / 25 corregimientos / Distrito de Panamá |
| INEC 2023 | 21,801 accidentes (YoY +6.6%) |
| FEDPA | 517 reclamos reales deidentificados (broker panameño) |
| Base del modelo | Florida Accidents Dataset (US_Accidents_FL.csv) |
| Open-Meteo API | Datos meteorológicos históricos para enriquecimiento del dataset |

### Ejecutar el Dashboard

```bash
cd notebooks/proyecto_integrador_3/dashboard
pip install -r requirements.txt
streamlit run app.py
```

Abre en: http://localhost:8501

> **Nota**: El archivo `accident_prediction_system.joblib` (~6.2 GB) **no está incluido** en el repositorio por su tamaño. Ejecútalo desde `proyecto_final.ipynb` en Colab o solicítalo al equipo.

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

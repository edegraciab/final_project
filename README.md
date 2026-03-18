# 🚗 Modelo Predictivo de Accidentes Automovilísticos

## Universidad Tecnológica de Panamá — Maestría en Analítica de Datos
### Proyectos Integradores I y II — 2026

---

## 📋 Descripción del Proyecto

Este repositorio contiene el desarrollo de un **modelo predictivo de accidentes automovilísticos** que utiliza el dataset **US Accidents** filtrado para el **estado de Florida**. El proyecto se desarrolla en dos fases académicas secuenciales, con especial énfasis en el **Proyecto Integrador II** (materia actual).

### 🎯 Objetivo Principal
Desarrollar un sistema predictivo de dos etapas para:
1. **Estimar la probabilidad de ocurrencia** de accidentes por zona geográfica
2. **Predecir el nivel de severidad** de accidentes (Low / Moderate / High)

### 🌴 Justificación de Florida
Florida fue seleccionado como estado de estudio debido a su **similitud climática y estructural con Panamá**:
- Clima subtropical
- Infraestructura mixta urbana/rural
- Geografía costera
- Patrones de tráfico similares

---

## 🗂️ Estructura del Proyecto

```
notebooks/
├── data/
│   ├── US_Accidents_FL.csv                    # Dataset filtrado de Florida
│   └── US_Accidents_encoded.csv               # Dataset procesado y codificado
├── proyecto_ingrador_1/                        # 📚 PROYECTO INTEGRADOR I
│   ├── 01.download_dataset.ipynb              # Descarga inicial del dataset
│   ├── 02.DASK_EDA.ipynb                      # Demo EDA con DASK para para pruebas big data
│   ├── 02.EDA.ipynb                           # Análisis exploratorio básico
│   └── 03.EDA_FL.ipynb                        # EDA específico de Florida
└── proyecto_integrador_2/                     # 🎯 PROYECTO INTEGRADOR II (ACTUAL)
    └── 04.EDA_FL_v2.ipynb                     # EDA avanzado y modelado predictivo
```

---

## 📈 Proyecto Integrador II (Enfoque Principal)

### 🔬 Características del Análisis Avanzado

El **Proyecto Integrador II** representa la culminación del análisis y se enfoca en:

#### 📊 Análisis Exploratorio Avanzado
- **Granularidad geográfica**: County y City level
- **Patrones temporales**: Análisis por hora, día, mes, estación
- **Condiciones ambientales**: Weather, visibility, temperature
- **Infraestructura vial**: Traffic signals, junctions, road features


#### 🤖 Modelado Predictivo
- **Enfoque de dos etapas**:
  1. **Modelo de Ocurrencia**: Probabilidad de que ocurra un accidente en una zona específica
  2. **Modelo de Severidad**: Clasificación del nivel de gravedad del accidente

#### 📋 Variables Clave Analizadas
- **Geográficas**: County, City, Start_Lat, Start_Lng
- **Temporales**: Hour, Day, Month, Season
- **Climatológicas**: Weather_Condition, Temperature, Humidity, Visibility
- **Infraestructura**: Traffic_Signal, Junction, Crossing, Railway
- **Severidad**: Target variable (Low/Moderate/High)

### 🛠️ Tecnologías y Librerías Utilizadas
- **Python 3.x**
- **Pandas & NumPy**: Manipulación de datos
- **Matplotlib & Seaborn**: Visualización estadística
- **Plotly**: Visualizaciones interactivas
- **Scikit-learn**: Modelado predictivo
- **Jupyter Notebooks**: Desarrollo interactivo

---

## 📚 Proyecto Integrador I (Fundacional)

### 🗃️ Notebooks Incluidos

#### [01.download_dataset.ipynb](notebooks/proyecto_ingrador_1/01.download_dataset.ipynb)
- **Propósito**: Descarga automática del dataset US Accidents desde Kaggle
- **Herramientas**: KaggleHub API
- **Output**: Dataset crudo almacenado localmente

#### [02.EDA.ipynb](notebooks/proyecto_ingrador_1/02.EDA.ipynb) & [02.DASK_EDA.ipynb](notebooks/proyecto_ingrador_1/02.DASK_EDA.ipynb)
- **Propósito**: Demo de Análisis exploratorio inicial del dataset completo 
- **Características**: 
  - Versión estándar con Pandas
  - Versión optimizada con DASK para big data
- **Insights**: Estadísticas descriptivas, distribuciones, calidad de datos

#### [03.EDA_FL.ipynb](notebooks/proyecto_ingrador_1/03.EDA_FL.ipynb)
- **Propósito**: Filtrado y análisis específico para Florida
- **Características**:
  - Filtrado geográfico por estado
  - Análisis de patrones específicos de Florida
  - Preparación para análisis avanzado

---

## 🚀 Cómo Ejecutar el Proyecto

### 📋 Prerrequisitos
```bash
# Instalar dependencias
pip install pandas numpy matplotlib seaborn plotly scikit-learn kagglehub jupyter
```

### 🔄 Flujo de Ejecución Recomendado

#### Para Proyecto Integrador I:
1. `01.download_dataset.ipynb` - Descargar datos
2. `02.EDA.ipynb`  - EDA inicial que genera el .csv filtrado para Florida
3. `03.EDA_FL.ipynb` - Filtrado a Florida para proyecto integrador 1

#### Para Proyecto Integrador II (Actual):
1. `04.EDA_FL_v2.ipynb` - **NOTEBOOK PRINCIPAL**
   - Análisis completo y avanzado
   - Desarrollo de modelos predictivos
   - Visualizaciones interactivas
   - Insights para la toma de decisiones
   - Enfoque en story telling

### 📊 Dataset
- **Fuente**: [US Accidents Dataset](https://www.kaggle.com/sobhanmoosavi/us-accidents) (Kaggle)
- **Tamaño Original**: ~3M registros
- **Tamaño Filtrado (FL)**: ~200K registros
- **Período**: 2016-2023
- **Features**: 46+ columnas

---


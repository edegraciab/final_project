"""
Sistema Predictivo de Accidentes — Demo Comercial Panamá
Calibrado con INEC 2023-2024 + FEDPA broker data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pickle
import json
import datetime
from pathlib import Path
import warnings
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from weather_enrichment import get_live_weather, build_climatology, wmo_to_string
warnings.filterwarnings("ignore")

# Import the class so it lives in this process's namespace
from model import AccidentPredictionSystem  # noqa: F401

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskMap PA · Sistema Predictivo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

:root {
    --red:     #C0392B;
    --red-dim: #922B21;
    --amber:   #E67E22;
    --green:   #27AE60;
    --blue:    #2980B9;
    --bg:      #0D0F14;
    --bg2:     #13161D;
    --bg3:     #1A1E28;
    --border:  #252A38;
    --text:    #E8ECF4;
    --muted:   #7A8499;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiSelect label { color: var(--muted) !important; font-size:11px !important; }

h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }
p, li, span, div { font-family: 'Inter', sans-serif !important; }
code, .mono { font-family: 'IBM Plex Mono', monospace !important; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
    color: var(--text);
    margin: 0;
}
.hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 6px;
}
.accent { color: var(--red); }

.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1.5rem 0; }
.kpi {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi.red::before   { background: var(--red); }
.kpi.amber::before { background: var(--amber); }
.kpi.green::before { background: var(--green); }
.kpi.blue::before  { background: var(--blue); }
.kpi-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-family: 'IBM Plex Mono', monospace; margin-bottom: 6px; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; line-height: 1; }
.kpi-sub   { font-size: 11px; color: var(--muted); margin-top: 4px; }

.risk-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
}
.pill-critico  { background: rgba(192,57,43,0.18); color: #E74C3C; border: 1px solid rgba(192,57,43,0.4); }
.pill-alto     { background: rgba(230,126,34,0.18); color: #F39C12; border: 1px solid rgba(230,126,34,0.4); }
.pill-moderado { background: rgba(39,174,96,0.18);  color: #2ECC71; border: 1px solid rgba(39,174,96,0.4); }
.pill-bajo     { background: rgba(41,128,185,0.18); color: #3498DB; border: 1px solid rgba(41,128,185,0.4); }

.pred-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.pred-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); font-family: 'IBM Plex Mono', monospace; }
.pred-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; margin-top: 4px; }
.bar-track { background: var(--border); border-radius: 4px; height: 6px; margin-top: 8px; }
.bar-fill  { height: 6px; border-radius: 4px; transition: width 0.4s ease; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

.prima-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.prima-table th {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--muted); padding: 8px 12px;
    border-bottom: 1px solid var(--border); text-align: left;
}
.prima-table td { padding: 9px 12px; border-bottom: 1px solid rgba(37,42,56,0.6); }
.prima-table tr:last-child td { border-bottom: none; }
.prima-table tr:hover td { background: var(--bg3); }

[data-baseweb="select"] > div {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
.stSlider [data-baseweb="slider"] { margin-top: 4px; }
div[data-testid="stTabs"] [role="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ─── Paths ─────────────────────────────────────────────────────────────────
_DASHBOARD_DIR   = Path(__file__).parent
_OUTPUT_DATA_DIR = _DASHBOARD_DIR.parents[1] / "output" / "data"
_FL_CSV_PATH     = _DASHBOARD_DIR.parents[1] / "data" / "US_Accidents_FL.csv"

# ─── Data & model ──────────────────────────────────────────────────────────
@st.cache_data
def load_fl_data(n: int = 60_000):
    """Carga una muestra estratificada del dataset de Florida para EDA. Si n=None, carga toda la data."""
    fl = pd.read_csv(_FL_CSV_PATH, parse_dates=["Start_Time", "End_Time"])
    fl["Start_Time"] = pd.to_datetime(fl["Start_Time"], errors="coerce")
    fl["End_Time"]   = pd.to_datetime(fl["End_Time"], errors="coerce")
    
    # Calcular duración real en minutos
    fl["Duration_min"] = (fl["End_Time"] - fl["Start_Time"]).dt.total_seconds() / 60.0
    
    # Clasificación MUTCD por tiempo real (FHWA)
    def clasificar_mutcd(duracion):
        if pd.isna(duracion) or duracion < 30:
            return "Menor"
        elif duracion <= 120:
            return "Intermedio"
        else:
            return "Mayor"
            
    fl["MUTCD"] = fl["Duration_min"].apply(clasificar_mutcd)
    
    if n is None or n >= len(fl):
        fl_sample = fl.copy()
    else:
        # Muestra estratificada por MUTCD
        fl_sample = (
            fl.groupby("MUTCD", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), int(n * len(g) / len(fl))), random_state=42))
        ).reset_index(drop=True)
        
    fl_sample["Hour"]      = fl_sample["Start_Time"].dt.hour
    fl_sample["DayOfWeek"] = fl_sample["Start_Time"].dt.dayofweek
    fl_sample["Month"]     = fl_sample["Start_Time"].dt.month
    fl_sample["Year"]      = fl_sample["Start_Time"].dt.year
    
    return fl, fl_sample

@st.cache_data
def load_data():
    df = pd.read_csv(_OUTPUT_DATA_DIR / "panama_synthetic_accidents_weather.csv")
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    df["wet_season"] = df["Month"].isin([5,6,7,8,9,10,11]).map({True:"Húmeda", False:"Seca"})
    bins = [0, 454, 648, 1969, 9999]
    labels = ["Bajo","Moderado","Alto","Crítico"]
    df["nivel_riesgo"] = pd.cut(df["INEC_2024"], bins=bins, labels=labels)
    return df

@st.cache_data
def load_road_dist():
    with open(_OUTPUT_DATA_DIR / "inec_road_dist.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_climatology():
    """Pre-computa medianas ERA5 por zona×mes×hora. Carga una sola vez."""
    return build_climatology(str(_OUTPUT_DATA_DIR / "panama_synthetic_accidents_weather.csv"))

@st.cache_data(ttl=600)
def cached_live_weather(lat: float, lng: float, date_str: str, hour: int) -> dict:
    """Llama get_live_weather con cache de 10 min para evitar llamadas repetidas."""
    target_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour)
    clim = load_climatology()
    return get_live_weather(
        lat=lat, lng=lng, target_dt=target_dt,
        climatology=clim,
        cache_dir=str(_DASHBOARD_DIR / ".openmeteo_cache"),
    )

@st.cache_resource
def load_model():
    import joblib
    import io

    class _FixedUnpickler(pickle.Unpickler):
        """Redirect __main__.AccidentPredictionSystem → model.AccidentPredictionSystem.

        The .joblib was saved from a Colab notebook cell, so pickle recorded
        the class module as '__main__'. This unpickler intercepts that lookup
        and returns the correct class from the standalone model module instead.
        """
        def find_class(self, module, name):
            if name == "AccidentPredictionSystem":
                from model import AccidentPredictionSystem
                return AccidentPredictionSystem
            return super().find_class(module, name)

    # joblib files are numpy-extended pickle streams; we must let joblib handle
    # the file framing (compression, numpy mmap, etc.) but swap the unpickler.
    # The simplest approach: patch pickle.Unpickler at load time.
    _orig_unpickler = pickle.Unpickler
    pickle.Unpickler = _FixedUnpickler
    try:
        obj = joblib.load(_DASHBOARD_DIR / "accident_prediction_system.joblib")
    finally:
        pickle.Unpickler = _orig_unpickler  # always restore
    return obj

@st.cache_data
def zone_summary(df):
    return (df.groupby("County")
        .agg(
            lat=("Start_Lat","mean"), lng=("Start_Lng","mean"),
            total=("Target_Severity","count"),
            p_mayor=("Target_Severity", lambda x: (x==2).mean()),
            p_menor=("Target_Severity", lambda x: (x==0).mean()),
            p_inter=("Target_Severity", lambda x: (x==1).mean()),
            INEC_2024=("INEC_2024","first"),
            INEC_2023=("INEC_2023","first"),
            YoY=("YoY_growth","first"),
            INEC_weight=("INEC_weight","first"),
            nivel=("nivel_riesgo","first"),
        )
        .reset_index()
    )

df = load_data()
sistema = load_model()
zones_df = zone_summary(df)
clim_loaded = False  # se carga lazy al primer predict

with open(_OUTPUT_DATA_DIR / "panama_severity_dist.json", encoding="utf-8") as _sev_f:
    severity_dist = json.load(_sev_f)

# (Static actuarial index removed - moved to dynamic loop)

RISK_COLORS = {"Crítico":"#E74C3C","Alto":"#F39C12","Moderado":"#2ECC71","Bajo":"#3498DB"}

# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0 0 20px'>
      <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;color:#E8ECF4'>
        RISKMAP <span style='color:#C0392B'>PA</span>
      </div>
      <div style='font-family:IBM Plex Mono,monospace;font-size:9px;color:#7A8499;letter-spacing:.12em;text-transform:uppercase'>
        Sistema Predictivo · Beta
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Parámetros del escenario")

    corregimiento = st.selectbox("Corregimiento", sorted(df["County"].unique()))
    hora          = st.slider("Hora del siniestro", 0, 23, 17)
    fecha_sel     = st.date_input(
        "Fecha del siniestro",
        value=datetime.date.today(),
        min_value=datetime.date(2020, 1, 1),
        max_value=datetime.date.today() + datetime.timedelta(days=16),
        help="Fechas dentro de ±5 días usan pronóstico en tiempo real. Fechas lejanas usan climatología ERA5.",
    )
    mes = fecha_sel.month
    dow = fecha_sel.weekday()   # 0=Lunes … 6=Domingo

    st.markdown("---")
    st.markdown("#### Infraestructura vial")
    junction   = st.checkbox("Intersección", value=True)
    signal     = st.checkbox("Semáforo",     value=True)
    crossing   = st.checkbox("Cruce peatonal")
    roundabout = st.checkbox("Rotonda")

    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:9px;text-transform:uppercase;
                letter-spacing:.12em;color:#7A8499;margin-bottom:10px'>
      Modelo predictivo
    </div>
    <div style='background:#1A1E28;border:1px solid #252A38;border-radius:6px;
                padding:12px 14px;margin-bottom:8px'>
      <div style='font-family:Syne,sans-serif;font-size:.85rem;font-weight:700;
                  color:#E8ECF4;margin-bottom:2px'>Pipeline 2 Etapas</div>
      <div style='font-size:10px;color:#7A8499;font-family:IBM Plex Mono,monospace'>
        Poisson GLM → Random Forest
      </div>
    </div>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:8px'>
      <div style='background:#1A1E28;border:1px solid #252A38;border-radius:6px;padding:8px 10px;text-align:center'>
        <div style='font-size:8px;color:#7A8499;font-family:IBM Plex Mono,monospace;
                    text-transform:uppercase;letter-spacing:.08em'>Accuracy</div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#2ECC71'>60.9%</div>
        <div style='font-size:8px;color:#7A8499'>test set</div>
      </div>
      <div style='background:#1A1E28;border:1px solid #252A38;border-radius:6px;padding:8px 10px;text-align:center'>
        <div style='font-size:8px;color:#7A8499;font-family:IBM Plex Mono,monospace;
                    text-transform:uppercase;letter-spacing:.08em'>ROC AUC</div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#2ECC71'>0.794</div>
        <div style='font-size:8px;color:#7A8499'>weighted</div>
      </div>
      <div style='background:#1A1E28;border:1px solid #252A38;border-radius:6px;padding:8px 10px;text-align:center'>
        <div style='font-size:8px;color:#7A8499;font-family:IBM Plex Mono,monospace;
                    text-transform:uppercase;letter-spacing:.08em'>AUC Mayor</div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#F39C12'>0.836</div>
        <div style='font-size:8px;color:#7A8499'>clase fatal</div>
      </div>
      <div style='background:#1A1E28;border:1px solid #252A38;border-radius:6px;padding:8px 10px;text-align:center'>
        <div style='font-size:8px;color:#7A8499;font-family:IBM Plex Mono,monospace;
                    text-transform:uppercase;letter-spacing:.08em'>Gini</div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#F39C12'>0.671</div>
        <div style='font-size:8px;color:#7A8499'>🟢 Bueno</div>
      </div>
    </div>
    <div style='font-size:9px;color:#4A5568;font-family:IBM Plex Mono,monospace;line-height:1.6'>
      Train: US Accidents FL 2016–23<br>
      Cal: INEC 2023–2024 · FEDPA<br>
      CV: 5-fold · n=617,735
    </div>
    """, unsafe_allow_html=True)


# ─── Fetch weather automático ─────────────────────────────────────────────
zone_row = zones_df[zones_df["County"] == corregimiento].iloc[0]
sunrise  = "Day" if 6 <= hora <= 18 else "Night"

wx = cached_live_weather(
    lat      = float(zone_row["lat"]),
    lng      = float(zone_row["lng"]),
    date_str = fecha_sel.strftime("%Y-%m-%d"),
    hour     = hora,
)

# ─── Compute prediction ────────────────────────────────────────────────────
input_row = pd.DataFrame([{
    "Start_Lat":         zone_row["lat"],
    "Start_Lng":         zone_row["lng"],
    "City":              "Panama City",
    "County":            corregimiento,
    "Amenity":           False,
    "Bump":              False,
    "Crossing":          crossing,
    "Give_Way":          False,
    "Junction":          junction,
    "No_Exit":           False,
    "Railway":           False,
    "Roundabout":        roundabout,
    "Station":           False,
    "Stop":              False,
    "Traffic_Calming":   False,
    "Traffic_Signal":    signal,
    "Temperature(F)":    wx["temperature_f"],
    "Humidity(%)":       wx["humidity_pct"],
    "Visibility(mi)":    wx["visibility_mi"],
    "Wind_Speed(mph)":   wx["wind_mph"],
    "Precipitation(in)": wx["precip_in"],
    "Hour":              hora,
    "DayOfWeek":         dow,
    "Month":             mes,
    "Weather_Condition": wx["weather_condition"],
    "Sunrise_Sunset":    sunrise,
}])

result = sistema.predict_severity(input_row)
#st.write("DEBUG columns:", result.columns.tolist())
#st.write("DEBUG result:", result)
#st.stop()  # detiene la ejecución aquí
probs  = result[["prob_Menor","prob_Intermedio","prob_Mayor"]].values[0]

# ── Etapa 1: Frecuencia esperada (Poisson) Dinámica ─────────────────────────
# Generamos el dataset Poisson para TODAS las zonas
poisson_rows = []
for c in zones_df["County"]:
    poisson_rows.append({
        "County":          c,
        "Hour":            hora,
        "Month":           mes,
        "temp_mean":       wx["temperature_f"],
        "humidity_mean":   wx["humidity_pct"],
        "visibility_mean": wx.get("visibility_mi", 7.0),
        "rain_mean":       wx["precip_in"],
    })
all_zones_poisson = pd.DataFrame(poisson_rows)

zones_dynamic = zones_df.copy()

try:
    freq_preds = sistema.predict_frequency(all_zones_poisson)
    zones_dynamic["freq_pred_dyn"] = np.maximum(0.0, np.round(freq_preds, 4))
except Exception:
    zones_dynamic["freq_pred_dyn"] = (zones_dynamic["INEC_2024"] / 8760).round(4)

freq_pred = float(zones_dynamic.loc[zones_dynamic["County"] == corregimiento, "freq_pred_dyn"].values[0])

# ── Índice Actuarial Dinámico ──────────────────────────────────────────
zones_dynamic["prima_index_dyn"] = (
    zones_dynamic["freq_pred_dyn"] * zones_dynamic["p_mayor"] * (1 + zones_dynamic["YoY"])
)
_idx_min = zones_dynamic["prima_index_dyn"].replace(0, np.nan).min()
zones_dynamic["prima_index_dyn"] = (
    (zones_dynamic["prima_index_dyn"] / max(_idx_min, 0.0001))
    .fillna(1.0)
    .clip(lower=0.01)
    .round(2)
)

def calc_nivel_dinamico(pi):
    if pi >= 5: return "Crítico"
    elif pi >= 2: return "Alto"
    elif pi >= 1: return "Moderado"
    else: return "Bajo"
    
zones_dynamic["nivel_dyn"] = zones_dynamic["prima_index_dyn"].apply(calc_nivel_dinamico)

# Fila del corregimiento seleccionado en el dataframe dinámico
zone_row_dyn = zones_dynamic[zones_dynamic["County"] == corregimiento].iloc[0]

# Calibrar probabilidad Mayor (Predictor local) usando la Frecuencia Dinámica
mean_f  = zones_dynamic["freq_pred_dyn"].mean()
cal_fac = (freq_pred / max(mean_f, 0.0001)) if mean_f > 0 else 1.0
prob_m  = min(probs[0], 1.0)
prob_i  = min(probs[1], 1.0)
prob_ma = min(probs[2] * cal_fac, 1.0)
# Renormalize
total_p = prob_m + prob_i + prob_ma
prob_m  /= total_p; prob_i /= total_p; prob_ma /= total_p

sev_idx   = int(np.argmax([prob_m, prob_i, prob_ma]))
sev_label = ["MENOR","INTERMEDIO","MAYOR"][sev_idx]
sev_color = ["#3498DB","#F39C12","#E74C3C"][sev_idx]

# ─── Main layout ──────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:8px'>
  <p class='hero-sub'>República de Panamá · Distrito Capital · 2024</p>
  <h1 class='hero-title'>Sistema Predictivo de<br><span class='accent'>Accidentes de Tránsito</span></h1>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──
total_acc = int(zones_df["INEC_2024"].sum())
yoy_dist  = 6.6
top_zone  = zones_df.loc[zones_df["INEC_2024"].idxmax(), "County"]
pct_mayor = float(df["Target_Severity"].eq(2).mean() * 100)

st.markdown(f"""
<div class='kpi-grid'>
  <div class='kpi red'>
    <div class='kpi-label'>Accidentes 2024 · INEC</div>
    <div class='kpi-value' style='color:#E74C3C'>{total_acc:,}</div>
    <div class='kpi-sub'>Distrito de Panamá</div>
  </div>
  <div class='kpi amber'>
    <div class='kpi-label'>Crecimiento YoY</div>
    <div class='kpi-value' style='color:#F39C12'>+{yoy_dist}%</div>
    <div class='kpi-sub'>vs 21,801 en 2023</div>
  </div>
  <div class='kpi blue'>
    <div class='kpi-label'>Zona más crítica</div>
    <div class='kpi-value' style='color:#3498DB;font-size:1.4rem'>{top_zone}</div>
    <div class='kpi-sub'>2,881 accidentes</div>
  </div>
  <div class='kpi green'>
    <div class='kpi-label'>Accidentes Mayor</div>
    <div class='kpi-value' style='color:#2ECC71'>{pct_mayor:.1f}%</div>
    <div class='kpi-sub'>Clasificación MUTCD</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──
tab_eda_fl, tab_mapa, tab_perfil, tab_predictor, tab_actuarial = st.tabs([
    "🔍  EDA · Florida",
    "🗺️  Mapa de Riesgo",
    "📊  Perfil de Siniestralidad",
    "🎯  Predictor",
    "💰  Actuarial"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════
with tab_predictor:
    col_pred, col_ctx = st.columns([1, 1.6], gap="large")

    with col_pred:
        st.markdown("<div class='section-header'>Predicción de severidad MUTCD</div>", unsafe_allow_html=True)

        nivel_str = str(zone_row["nivel"])
        pill_cls  = {"Crítico":"pill-critico","Alto":"pill-alto","Moderado":"pill-moderado","Bajo":"pill-bajo"}.get(nivel_str, "pill-bajo")

        st.markdown(f"""
        <div class='pred-card'>
          <div class='pred-label'>Corregimiento seleccionado</div>
          <div class='pred-value'>{corregimiento}</div>
          <div style='margin-top:8px'>
            <span class='risk-pill {pill_cls}'>{nivel_str}</span>
            <span style='font-size:12px;color:#7A8499;margin-left:8px'>INEC 2024: {int(zone_row["INEC_2024"]):,}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='pred-card' style='border-color:{sev_color}40'>
          <div class='pred-label'>Severidad predicha</div>
          <div class='pred-value' style='color:{sev_color}'>{sev_label}</div>
          <div style='font-size:11px;color:#7A8499;margin-top:4px'>
            Hora {hora:02d}:00 · {"Wet" if mes in [5,6,7,8,9,10,11] else "Dry"} season · {wx["weather_condition"]}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Etapa 1 — Frecuencia esperada Poisson ──
        st.markdown("<div class='section-header'>Etapa 1 · Frecuencia esperada (Poisson)</div>",
                    unsafe_allow_html=True)

        freq_color = (
            "#E74C3C" if freq_pred >= 3.0
            else "#F39C12" if freq_pred >= 1.5
            else "#2ECC71"
        )
        freq_label = (
            "Alta frecuencia" if freq_pred >= 3.0
            else "Frecuencia moderada" if freq_pred >= 1.5
            else "Baja frecuencia"
        )

        st.markdown(f"""
        <div class='pred-card' style='margin-bottom:10px'>
          <div class='pred-label'>Accidentes esperados / hora · zona · condición</div>
          <div style='display:flex;align-items:baseline;gap:10px;margin-top:4px'>
            <div class='pred-value' style='color:{freq_color}'>{freq_pred:.2f}</div>
            <div style='font-size:12px;color:#7A8499'>acc/hora</div>
          </div>
          <div style='margin-top:8px;display:flex;align-items:center;gap:8px'>
            <span class='risk-pill' style='background:{freq_color}22;
                  color:{freq_color};border:1px solid {freq_color}44'>
              {freq_label}
            </span>
            <span style='font-size:11px;color:#7A8499'>
              {corregimiento} · {hora:02d}:00h · {wx["weather_condition"]}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Etapa 2 — Severidad condicional (Random Forest) ──
        st.markdown("<div class='section-header'>Etapa 2 · Severidad condicional (Random Forest)</div>", unsafe_allow_html=True)

        for label, prob, color in [("MENOR",prob_m,"#3498DB"),("INTERMEDIO",prob_i,"#F39C12"),("MAYOR",prob_ma,"#E74C3C")]:
            st.markdown(f"""
            <div style='margin-bottom:10px'>
              <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px'>
                <span style='font-family:IBM Plex Mono,monospace;color:#7A8499'>{label}</span>
                <span style='font-weight:600;color:{color}'>{prob*100:.1f}%</span>
              </div>
              <div class='bar-track'>
                <div class='bar-fill' style='width:{prob*100:.1f}%;background:{color}'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Prima técnica combinada (Etapa 1 × Etapa 2) ──
        prima_combinada = freq_pred * prob_ma
        st.markdown(f"""
        <div style='background:#13161D;border:1px solid #252A38;border-radius:8px;
                    padding:12px 16px;margin-bottom:12px'>
          <div style='font-family:IBM Plex Mono,monospace;font-size:9px;text-transform:uppercase;
                      letter-spacing:.12em;color:#7A8499;margin-bottom:6px'>
            E\u005bFrec\u005d \u00d7 P(Mayor) \u2192 exposici\u00f3n actuarial
          </div>
          <div style='display:flex;align-items:baseline;gap:8px'>
            <span style='font-family:Syne,sans-serif;font-size:1.5rem;
                         font-weight:700;color:#F39C12'>{prima_combinada:.4f}</span>
            <span style='font-size:11px;color:#7A8499'>acc. mayores esperados/hora</span>
          </div>
          <div style='font-size:10px;color:#4A5568;font-family:IBM Plex Mono,monospace;margin-top:4px'>
            {freq_pred:.2f} acc/h \u00d7 {prob_ma*100:.2f}% P(Mayor) 
            \u00d7 INEC_weight {float(zone_row["INEC_weight"]):.4f}
          </div>
        </div>
        """, unsafe_allow_html=True)

        prima_idx = float(zone_row_dyn["prima_index_dyn"])
        yoy_pct   = float(zone_row_dyn["YoY"]) * 100
        st.markdown(f"""
        <div class='pred-card' style='margin-top:12px;background:#13161D'>
          <div class='pred-label'>Índice de prima técnica relativa</div>
          <div class='pred-value' style='color:#F39C12'>{prima_idx:.2f}×</div>
          <div style='font-size:11px;color:#7A8499;margin-top:4px'>
            Crecimiento YoY zona: <span style='color:{"#E74C3C" if yoy_pct>0 else "#2ECC71"}'>{yoy_pct:+.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ctx:
        st.markdown("<div class='section-header'>Contexto climático e infraestructura</div>", unsafe_allow_html=True)

        # ── Tarjeta de condiciones climáticas automáticas ──
        _src        = wx.get("source", "default")
        _src_badge  = {
            "forecast":    ("🟢", "En vivo · Forecast",   "#27AE60"),
            "climatology": ("📊", "ERA5 · Climatología",   "#2980B9"),
            "default":     ("⚪", "Valores por defecto",    "#7A8499"),
        }.get(_src, ("⚪", _src, "#7A8499"))
        _wmo_icons = {
            "Clear":"☀️","Mostly Clear":"🌤️","Partly Cloudy":"⛅",
            "Overcast":"☁️","Fog":"🌫️","Light Drizzle":"🌦️",
            "Drizzle":"🌦️","Heavy Drizzle":"🌧️","Light Rain":"🌧️",
            "Rain":"🌧️","Heavy Rain":"🌧️","Rain Showers":"🌧️",
            "Heavy Rain Showers":"⛈️","Thunderstorm":"⛈️",
        }
        _wicon = _wmo_icons.get(wx["weather_condition"], "🌡️")
        st.markdown(f"""
        <div style='background:#13161D;border:1px solid #252A38;border-radius:10px;
                    padding:14px 18px;margin-bottom:14px'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>
            <div style='font-family:IBM Plex Mono,monospace;font-size:9px;
                        text-transform:uppercase;letter-spacing:.12em;color:#7A8499'>
              Condiciones climáticas
            </div>
            <div style='font-size:10px;color:{_src_badge[2]};font-family:IBM Plex Mono,monospace;
                        background:{_src_badge[2]}22;border:1px solid {_src_badge[2]}44;
                        border-radius:20px;padding:2px 10px'>
              {_src_badge[0]} {_src_badge[1]}
            </div>
          </div>
          <div style='font-size:1.6rem;margin-bottom:8px'>{_wicon}
            <span style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
                         color:#E8ECF4;margin-left:6px'>{wx["weather_condition"]}</span>
          </div>
          <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px'>
            <div style='background:#1A1E28;border-radius:6px;padding:8px 10px;text-align:center'>
              <div style='font-size:8px;color:#7A8499;font-family:IBM Plex Mono,monospace;
                          text-transform:uppercase;letter-spacing:.08em'>Temp</div>
              <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                          color:#F39C12'>{wx["temperature_f"]:.1f}°F</div>
            </div>
            <div style='background:#1A1E28;border-radius:6px;padding:8px 10px;text-align:center'>
              <div style='font-size:8px;color:#7A8499;font-family:IBM Plex Mono,monospace;
                          text-transform:uppercase;letter-spacing:.08em'>Humedad</div>
              <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                          color:#3498DB'>{wx["humidity_pct"]:.0f}%</div>
            </div>
            <div style='background:#1A1E28;border-radius:6px;padding:8px 10px;text-align:center'>
              <div style='font-size:8px;color:#7A8499;font-family:IBM Plex Mono,monospace;
                          text-transform:uppercase;letter-spacing:.08em'>Precip</div>
              <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                          color:#{'E74C3C' if wx['precip_in']>0.1 else '7A8499'}'>{wx["precip_in"]:.3f}"</div>
            </div>
            <div style='background:#1A1E28;border-radius:6px;padding:8px 10px;text-align:center'>
              <div style='font-size:8px;color:#7A8499;font-family:IBM Plex Mono,monospace;
                          text-transform:uppercase;letter-spacing:.08em'>Viento</div>
              <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                          color:#E8ECF4'>{wx["wind_mph"]:.1f} mph</div>
            </div>
            <div style='background:#1A1E28;border-radius:6px;padding:8px 10px;text-align:center'>
              <div style='font-size:8px;color:#7A8499;font-family:IBM Plex Mono,monospace;
                          text-transform:uppercase;letter-spacing:.08em'>Ráfagas</div>
              <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                          color:#E8ECF4'>{wx.get("gusts_mph", 0):.1f} mph</div>
            </div>
            <div style='background:#1A1E28;border-radius:6px;padding:8px 10px;text-align:center'>
              <div style='font-size:8px;color:#7A8499;font-family:IBM Plex Mono,monospace;
                          text-transform:uppercase;letter-spacing:.08em'>Nubosidad</div>
              <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                          color:#E8ECF4'>{wx.get("cloud_pct", 0):.0f}%</div>
            </div>
          </div>
          <div style='margin-top:10px;font-size:9px;color:#4A5568;font-family:IBM Plex Mono,monospace'>
            {fecha_sel.strftime("%d %b %Y")} · {hora:02d}:00 h · {corregimiento}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart — prob_mayor
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob_ma * 100, 1),
            title={"text": "P(Accidente Mayor) calibrada", "font": {"size": 13, "color": "#7A8499", "family": "IBM Plex Mono"}},
            number={"suffix": "%", "font": {"size": 32, "color": "#E74C3C", "family": "Syne"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#252A38", "tickfont": {"color": "#7A8499", "size": 10}},
                "bar": {"color": "#C0392B"},
                "bgcolor": "#1A1E28",
                "bordercolor": "#252A38",
                "steps": [
                    {"range": [0, 15], "color": "#1A2B1E"},
                    {"range": [15, 30], "color": "#2B2514"},
                    {"range": [30, 100], "color": "#2B1414"},
                ],
                "threshold": {"line": {"color": "#E74C3C", "width": 2}, "thickness": 0.75, "value": 25}
            }
        ))
        fig_gauge.update_layout(
            height=220, margin=dict(l=20,r=20,t=40,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8ECF4"
        )
        st.plotly_chart(fig_gauge, width='stretch')

        # Mini map for selected zone
        m = folium.Map(
            location=[zone_row["lat"], zone_row["lng"]],
            zoom_start=13,
            tiles="CartoDB dark_matter"
        )
        folium.CircleMarker(
            location=[zone_row["lat"], zone_row["lng"]],
            radius=18,
            color=RISK_COLORS.get(str(zone_row["nivel"]), "#3498DB"),
            fill=True, fill_opacity=0.5, weight=2,
            popup=folium.Popup(f"<b>{corregimiento}</b><br>INEC 2024: {int(zone_row['INEC_2024']):,}", max_width=160),
        ).add_to(m)
        # Sample points in zone
        zone_pts = df[df["County"] == corregimiento].sample(min(80, len(df[df["County"]==corregimiento])), random_state=42)
        for _, row in zone_pts.iterrows():
            c = ["#3498DB","#F39C12","#E74C3C"][int(row["Target_Severity"])]
            folium.CircleMarker(
                location=[row["Start_Lat"], row["Start_Lng"]],
                radius=3, color=c, fill=True, fill_opacity=0.6, weight=0
            ).add_to(m)
        st_folium(m, height=280, width='stretch')

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — MAPA DE RIESGO
# ═══════════════════════════════════════════════════════════════════════════
with tab_mapa:
    col_map, col_legend = st.columns([2.5, 1])

    with col_map:
        st.markdown("<div class='section-header'>Mapa de calor de accidentalidad — Distrito de Panamá 2024</div>", unsafe_allow_html=True)

        m2 = folium.Map(location=[9.02, -79.52], zoom_start=11, tiles="CartoDB dark_matter")

        for _, row in zones_dynamic.iterrows():
            nivel = str(row["nivel_dyn"])
            color = RISK_COLORS.get(nivel, "#3498DB")
            radius = min(40, int(row["freq_pred_dyn"] * 50) + 8)
            yoy_pct = row["YoY"] * 100

            folium.CircleMarker(
                location=[row["lat"], row["lng"]],
                radius=radius,
                color=color, fill=True, fill_opacity=0.55, weight=1.5,
                popup=folium.Popup(
                    f"""<div style='font-family:sans-serif;font-size:12px'>
                    <b>{row['County']}</b><br>
                    Nivel: <b>{nivel}</b><br>
                    INEC 2024: <b>{int(row['INEC_2024']):,}</b><br>
                    Crec. YoY: <b>{yoy_pct:+.1f}%</b><br>
                    P(Mayor): <b>{row['p_mayor']*100:.1f}%</b><br>
                    E[Freq] Poisson: <b>{row['freq_pred_dyn']:.3f}</b><br>
                    Prima index: <b>{row['prima_index_dyn']:.2f}×</b>
                    </div>""",
                    max_width=200
                ),
                tooltip=f"{row['County']} · {int(row['INEC_2024']):,} acc."
            ).add_to(m2)

        st_folium(m2, height=520, width='stretch')

    with col_legend:
        st.markdown("<div class='section-header'>Leyenda Dinámica</div>", unsafe_allow_html=True)
        for nivel, color in RISK_COLORS.items():
            cnt = zones_dynamic[zones_dynamic["nivel_dyn"].astype(str) == nivel]["County"].count()
            rng_map = {"Crítico":"1,970–2,881","Alto":"649–1,969","Moderado":"455–648","Bajo":"36–454"}
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px'>
              <div style='width:12px;height:12px;border-radius:50%;background:{color};flex-shrink:0'></div>
              <div>
                <div style='font-size:12px;font-weight:600'>{nivel}</div>
                <div style='font-size:10px;color:#7A8499;font-family:IBM Plex Mono,monospace'>{rng_map[nivel]} · {cnt} zonas</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='margin-top:24px'>Top 5 Zonas (Riesgo Actual)</div>", unsafe_allow_html=True)
        top5 = zones_dynamic.nlargest(5, "prima_index_dyn")
        for _, r in top5.iterrows():
            nv = str(r["nivel_dyn"])
            c  = RISK_COLORS.get(nv, "#3498DB")
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:8px 0;border-bottom:1px solid #252A38;font-size:12px'>
              <div>
                <div style='font-weight:500'>{r['County']}</div>
                <div style='font-size:10px;color:#7A8499'>YoY {r['YoY']*100:+.1f}%</div>
              </div>
              <div style='font-family:Syne,sans-serif;font-weight:700;color:{c}'>{r['prima_index_dyn']:.2f}×</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — ANÁLISIS
# ═══════════════════════════════════════════════════════════════════════════
with tab_perfil:
    # Synthetic charts moved below
    st.markdown("<div class='section-header'>Comparativo YoY por corregimiento — INEC 2023 vs 2024</div>", unsafe_allow_html=True)
    yoy_df = zones_df[["County","INEC_2023","INEC_2024","nivel"]].melt(
        id_vars=["County","nivel"], var_name="Año", value_name="Accidentes"
    )
    fig_yoy = px.bar(yoy_df, x="County", y="Accidentes", color="Año",
        barmode="group",
        color_discrete_map={"INEC_2023":"#4A5568","INEC_2024":"#E74C3C"},
        labels={"County":"Corregimiento"},
        template="plotly_dark")
    fig_yoy.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
        margin=dict(l=0,r=0,t=10,b=40), height=320,
        legend_title_text="", legend=dict(orientation="h", y=1.02),
        xaxis_tickangle=-40
    )
    fig_yoy.update_xaxes(showgrid=False)
    fig_yoy.update_yaxes(gridcolor="#252A38")
    st.plotly_chart(fig_yoy, width='stretch')

    # ── Concentración de Riesgo por Vía (Micro-segmentación) ─────────────
    st.markdown("<div class='section-header' style='margin-top:28px'>Concentración de Riesgo por Vía (Micro-segmentación)</div>", unsafe_allow_html=True)
    
    road_dist = load_road_dist()
    zone_vias = road_dist.get("zone_vias", {})
    
    available_counties = sorted([c for c in zones_df["County"].unique() if c in zone_vias])
    
    if available_counties:
        default_idx = available_counties.index("Ancón") if "Ancón" in available_counties else 0
        selected_county = st.selectbox("Seleccionar Corregimiento", available_counties, index=default_idx, key="road_dist_county")
        
        county_vias = zone_vias[selected_county]
        county_total = zones_df.loc[zones_df["County"] == selected_county, "INEC_2024"].values[0]
        
        _vias_df = pd.DataFrame({
            "Vía": county_vias["vias"],
            "Probabilidad": county_vias["probs"],
            "Tipo": county_vias["road_types"]
        })
        _vias_df["Accidentes_Estimados"] = (_vias_df["Probabilidad"] * county_total).round().astype(int)
        
        _vias_df = _vias_df.sort_values("Accidentes_Estimados", ascending=False).head(10)
        _vias_df = _vias_df.iloc[::-1]
        
        fig_roads = px.bar(_vias_df, x="Accidentes_Estimados", y="Vía", orientation="h",
                           text="Accidentes_Estimados",
                           color="Accidentes_Estimados", color_continuous_scale="Reds",
                           template="plotly_dark")
        fig_roads.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
            margin=dict(l=0,r=0,t=10,b=10), height=350,
            xaxis_title="Accidentes Estimados (INEC 2024)", yaxis_title="",
            showlegend=False, coloraxis_showscale=False
        )
        fig_roads.update_xaxes(showgrid=False)
        fig_roads.update_yaxes(gridcolor="#252A38")
        
        st.plotly_chart(fig_roads, width="stretch")
        
        top_via = _vias_df.iloc[-1]
        st.markdown(f"""
        <div style='font-size:13px;color:#7A8499;margin-top:-10px;margin-bottom:20px'>
            <b>Insight de micro-segmentación:</b> La vía más crítica en {selected_county} es <b>{top_via['Vía']}</b>, 
            concentrando el <b>{top_via['Probabilidad']*100:.1f}%</b> del riesgo total del corregimiento ({top_via['Accidentes_Estimados']} siniestros estimados en 2024).
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='margin-top:28px'>Severidad Actuarial por Tipo de Vía (Road_Type)</div>", unsafe_allow_html=True)
        
        _rt_df = df.groupby("Road_Type").apply(lambda x: pd.Series({
            "total": len(x),
            "p_mayor": (x["MUTCD_Category"]=="Mayor").mean() * 100
        }), include_groups=False).reset_index()
        _rt_df = _rt_df[_rt_df["total"] > 10].sort_values("p_mayor", ascending=True)
        
        fig_rt = px.bar(_rt_df, x="p_mayor", y="Road_Type", orientation="h",
                        text=_rt_df["p_mayor"].apply(lambda x: f"{x:.2f}%" if x > 0 else "&lt; 0.01%"),
                        color="p_mayor", color_continuous_scale="Reds",
                        template="plotly_dark")
        fig_rt.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
            margin=dict(l=0,r=0,t=10,b=10), height=250,
            xaxis_title="P(Mayor) %", yaxis_title="",
            showlegend=False, coloraxis_showscale=False
        )
        fig_rt.update_xaxes(showgrid=False)
        fig_rt.update_yaxes(gridcolor="#252A38")
        st.plotly_chart(fig_rt, width="stretch")

    # Synthetic charts moved below
    # ── Domain shift: mortalidad por tipo y picos estacionales ─────────────
    st.markdown("<div class='section-header' style='margin-top:28px'>Mortalidad por tipo de accidente · Picos estacionales — INEC Real Panamá</div>", unsafe_allow_html=True)

    c5, c6 = st.columns(2, gap="medium")

    with c5:
        st.markdown("<div class='section-header'>Tasa fatal por clase de accidente</div>", unsafe_allow_html=True)

        _clase_rows = [
            {"clase": k, "p_mayor_pct": v["p_mayor"] * 100}
            for k, v in severity_dist["by_clase"].items()
        ]
        _mort_df = pd.DataFrame(_clase_rows).sort_values("p_mayor_pct")

        _bar_mort_colors = [
            "#E74C3C" if p >= 5 else "#F39C12" if p >= 0.5 else "#4A5568"
            for p in _mort_df["p_mayor_pct"]
        ]

        fig_mort = go.Figure(go.Bar(
            x=_mort_df["p_mayor_pct"],
            y=_mort_df["clase"],
            orientation="h",
            marker_color=_bar_mort_colors,
            marker_line_width=0,
            text=[f"{p:.1f}%" for p in _mort_df["p_mayor_pct"]],
            textposition="outside",
            textfont=dict(size=10, color="#E8ECF4", family="IBM Plex Mono"),
            hovertemplate="<b>%{y}</b><br>P(fatal): %{x:.2f}%<extra></extra>"
        ))
        fig_mort.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
            margin=dict(l=0, r=55, t=10, b=10),
            xaxis=dict(
                title="P(fatal) %", gridcolor="#252A38",
                tickfont=dict(color="#7A8499", size=10)
            ),
            yaxis=dict(tickfont=dict(color="#E8ECF4", size=10)),
            showlegend=False,
            template="plotly_dark"
        )
        st.plotly_chart(fig_mort, width='stretch')

    with c6:
        st.markdown("<div class='section-header'>Accidentes fatales por mes — picos Jun · Dic</div>", unsafe_allow_html=True)

        _MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        _fatal_mo = (
            df[df["Target_Severity"] == 2]
            .groupby("Month").size()
            .reindex(range(1, 13), fill_value=0)
        )
        _total_mo = df.groupby("Month").size().reindex(range(1, 13), fill_value=1)
        _tasa_mo  = (_fatal_mo / _total_mo * 100).round(3)

        _pico_months = {6, 12}
        _bar_season_colors = [
            "#C0392B" if m in _pico_months else "#1F2D40"
            for m in range(1, 13)
        ]
        _bar_season_lines = [
            "#E74C3C" if m in _pico_months else "#2C3E50"
            for m in range(1, 13)
        ]

        fig_picos = go.Figure(go.Bar(
            x=_MESES,
            y=_tasa_mo.values,
            marker_color=_bar_season_colors,
            marker_line_color=_bar_season_lines,
            marker_line_width=1.5,
            text=[f"{t:.2f}%" if t > 0 else "" for t in _tasa_mo.values],
            textposition="outside",
            textfont=dict(size=9, color="#E8ECF4", family="IBM Plex Mono"),
            hovertemplate="<b>%{x}</b><br>Tasa fatal: %{y:.3f}%<br>Casos: %{customdata}<extra></extra>",
            customdata=_fatal_mo.values
        ))

        # Annotations for Jun and Dec peaks
        for _mi, _lbl in [(5, "Inicio\ntemporada"), (11, "Fiestas\nfin de año")]:
            _yval = _tasa_mo.values[_mi]
            fig_picos.add_annotation(
                x=_MESES[_mi],
                y=_yval,
                text=_lbl,
                showarrow=True,
                arrowhead=2, arrowcolor="#E74C3C", arrowsize=0.8,
                ay=-40, ax=0,
                font=dict(size=9, color="#E74C3C", family="IBM Plex Mono"),
                yanchor="top"
            )

        fig_picos.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(showgrid=False, tickfont=dict(color="#E8ECF4", size=11)),
            yaxis=dict(
                title="Tasa fatal (%)", gridcolor="#252A38",
                tickfont=dict(color="#7A8499", size=10)
            ),
            showlegend=False,
            template="plotly_dark"
        )
        st.plotly_chart(fig_picos, width='stretch')

    # ── Domain shift callout ───────────────────────────────────────────────
    st.markdown("""
    <div style='background:#13161D;border:1px solid rgba(192,57,43,0.35);border-radius:8px;
                padding:16px 20px;margin-top:6px;display:flex;align-items:center;gap:20px'>
      <div style='flex:1;font-size:12px;color:#E8ECF4;line-height:1.75'>
        <span style='font-family:IBM Plex Mono,monospace;font-size:9px;text-transform:uppercase;
                     letter-spacing:.12em;color:#C0392B'>⚠ Domain Shift FL → PA</span><br>
        El modelo fue entrenado con datos de <b>Florida</b> donde el
        <b style='color:#E74C3C'>17%</b> de accidentes son clasificados como Mayor.
        En Panamá real (INEC), solo el <b style='color:#2ECC71'>0.3%</b> son fatales
        — una diferencia de <b style='color:#F39C12'>~57×</b>.
        Sin corrección de este drift, los índices de prima sobreestiman el riesgo estructuralmente.
      </div>
      <div style='display:flex;gap:12px;flex-shrink:0'>
        <div style='text-align:center;padding:10px 16px;background:#1A1E28;border-radius:6px;
                    border:1px solid #252A38'>
          <div style='font-family:IBM Plex Mono,monospace;font-size:8px;color:#7A8499;
                      text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Atropello</div>
          <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:700;color:#E74C3C'>6.8%</div>
          <div style='font-size:9px;color:#7A8499'>tasa fatal</div>
        </div>
        <div style='text-align:center;padding:10px 16px;background:#1A1E28;border-radius:6px;
                    border:1px solid #252A38'>
          <div style='font-family:IBM Plex Mono,monospace;font-size:8px;color:#7A8499;
                      text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>Colisión</div>
          <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:700;color:#2ECC71'>0.08%</div>
          <div style='font-size:9px;color:#7A8499'>tasa fatal</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Distribuciones Calibradas ─────────────────────────────────────────
    st.markdown("<div class='section-header' style='margin-top:28px'>Distribución horaria y estacional — Calibradas con INEC Panamá</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("<div class='section-header'>Distribución horaria por severidad</div>", unsafe_allow_html=True)
        fig_hour = px.histogram(df, x="Hour", color="MUTCD_Category",
            barmode="overlay", opacity=0.75,
            category_orders={"MUTCD_Category":["Menor","Intermedio","Mayor"]},
            color_discrete_sequence=["#3498DB","#F39C12","#E74C3C"],
            labels={"Hour":"Hora","count":"Accidentes"},
            template="plotly_dark")
        fig_hour.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
            margin=dict(l=0,r=0,t=10,b=0), height=280, legend_title_text="",
            legend=dict(orientation="h", y=1.05)
        )
        fig_hour.update_xaxes(showgrid=False)
        fig_hour.update_yaxes(gridcolor="#252A38")
        st.plotly_chart(fig_hour, width='stretch')

    with c2:
        st.markdown("<div class='section-header'>Severidad por estación climática</div>", unsafe_allow_html=True)
        season_sev = df.groupby(["wet_season","MUTCD_Category"]).size().reset_index(name="n")
        fig_season = px.bar(season_sev, x="wet_season", y="n", color="MUTCD_Category",
            barmode="group",
            category_orders={"MUTCD_Category":["Menor","Intermedio","Mayor"]},
            color_discrete_sequence=["#3498DB","#F39C12","#E74C3C"],
            labels={"wet_season":"Estación","n":"Accidentes","MUTCD_Category":"Severidad"},
            template="plotly_dark")
        fig_season.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
            margin=dict(l=0,r=0,t=10,b=0), height=280, legend_title_text="",
            legend=dict(orientation="h", y=1.05)
        )
        fig_season.update_xaxes(showgrid=False)
        fig_season.update_yaxes(gridcolor="#252A38")
        st.plotly_chart(fig_season, width='stretch')

    st.markdown("<div class='section-header' style='margin-top:28px'>Mapa de Calor: Probabilidad Conjunta (Día × Hora) — INEC Panamá</div>", unsafe_allow_html=True)
    
    _joint_df = pd.read_csv(_OUTPUT_DATA_DIR / "inec_hour_dow_joint.csv")
    _dow_map = {0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}
    _joint_df["Día"] = _joint_df["dow"].map(_dow_map)
    _joint_df["Probabilidad (%)"] = _joint_df["prob"] * 100
    
    _hm_data = _joint_df.pivot(index="Día", columns="hour", values="Probabilidad (%)")
    _hm_data = _hm_data.reindex(["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"])
    
    fig_hm = px.imshow(_hm_data, 
                       labels=dict(x="Hora del día", y="Día de la semana", color="Prob (%)"),
                       x=_hm_data.columns, y=_hm_data.index,
                       color_continuous_scale="Reds",
                       aspect="auto",
                       template="plotly_dark")
    fig_hm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
        margin=dict(l=0,r=0,t=10,b=0), height=280
    )
    fig_hm.update_xaxes(tickmode='linear', dtick=1, showgrid=False)
    fig_hm.update_yaxes(showgrid=False)
    st.plotly_chart(fig_hm, width='stretch')




# ═══════════════════════════════════════════════════════════════════════════
# TAB EDA FL — ANÁLISIS EXPLORATORIO · FLORIDA
# ═══════════════════════════════════════════════════════════════════════════
with tab_eda_fl:
    fl_full, fl = load_fl_data(n=None)

    # ── KPIs ──────────────────────────────────────────────────────────────
    _fl_n       = len(fl_full)
    _fl_years   = f"{int(fl_full['Start_Time'].dt.year.min())}–{int(fl_full['Start_Time'].dt.year.max())}"
    _fl_top_city = fl_full["City"].value_counts().index[0]
    _fl_pct_major = (fl_full["MUTCD"] == "Mayor").mean() * 100

    st.markdown(f"""
    <div style='margin-bottom:16px'>
      <div style='font-family:IBM Plex Mono,monospace;font-size:9px;text-transform:uppercase;
                  letter-spacing:.15em;color:#7A8499;margin-bottom:6px'>
        Fuente de entrenamiento del modelo · US Accidents Dataset — Florida
      </div>
      <p style='font-size:13px;color:#E8ECF4;line-height:1.7;max-width:900px'>
        Este análisis exploratorio sobre los <b>{_fl_n:,}</b> registros de Florida ({_fl_years})
        identificó los patrones de hora, clima e infraestructura vial que luego
        se convirtieron en las features del modelo predictivo calibrado para Panamá.
        Las visualizaciones utilizan la totalidad de los registros de entrenamiento.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='kpi-grid'>
      <div class='kpi red'>
        <div class='kpi-label'>Registros totales FL</div>
        <div class='kpi-value' style='color:#E74C3C'>{_fl_n:,}</div>
        <div class='kpi-sub'>{_fl_years}</div>
      </div>
      <div class='kpi amber'>
        <div class='kpi-label'>Accidentes Mayor (> 2 hrs)</div>
        <div class='kpi-value' style='color:#F39C12'>{_fl_pct_major:.1f}%</div>
        <div class='kpi-sub'>Base de entrenamiento</div>
      </div>
      <div class='kpi blue'>
        <div class='kpi-label'>Ciudad más accidentada</div>
        <div class='kpi-value' style='color:#3498DB;font-size:1.3rem'>{_fl_top_city}</div>
        <div class='kpi-sub'>Florida</div>
      </div>
      <div class='kpi green'>
        <div class='kpi-label'>Features del modelo</div>
        <div class='kpi-value' style='color:#2ECC71'>22</div>
        <div class='kpi-sub'>Seleccionadas del EDA</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Fila 1: Severidad + Hora ───────────────────────────────────────────
    st.markdown("<div class='section-header'>Distribución de severidad y patrones temporales</div>",
                unsafe_allow_html=True)
    c_sev, c_hour = st.columns(2, gap="medium")

    with c_sev:
        st.markdown("<div class='section-header'>Distribución MUTCD (clasificación por tiempo real)</div>",
                    unsafe_allow_html=True)
        _sev_counts = fl["MUTCD"].value_counts().reindex(["Mayor","Intermedio","Menor"]).reset_index()
        _sev_counts.columns = ["Severidad","n"]
        _sev_colors = {"Mayor":"#E74C3C","Intermedio":"#F39C12","Menor":"#3498DB"}
        fig_sev = px.bar(
            _sev_counts, x="Severidad", y="n",
            color="Severidad",
            color_discrete_map=_sev_colors,
            text="n",
            template="plotly_dark",
            labels={"n":"Accidentes"},
        )
        fig_sev.update_traces(texttemplate="%{text:,}", textposition="outside",
                              textfont_size=10, textfont_color="#E8ECF4")
        fig_sev.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
            margin=dict(l=0,r=0,t=10,b=0), height=280,
            showlegend=False, xaxis_title=None,
        )
        fig_sev.update_yaxes(gridcolor="#252A38")
        fig_sev.update_xaxes(showgrid=False)
        st.plotly_chart(fig_sev, width="stretch")

    with c_hour:
        st.markdown("<div class='section-header'>Accidentes por hora del día</div>",
                    unsafe_allow_html=True)
        _hour_sev = fl.groupby(["Hour","MUTCD"]).size().reset_index(name="n")
        fig_hr = px.line(
            _hour_sev, x="Hour", y="n", color="MUTCD",
            color_discrete_map={"Mayor":"#E74C3C","Intermedio":"#F39C12","Menor":"#3498DB"},
            markers=True, template="plotly_dark",
            labels={"Hour":"Hora","n":"Accidentes","MUTCD":"Severidad"},
        )
        fig_hr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
            margin=dict(l=0,r=0,t=10,b=0), height=280,
            legend=dict(orientation="h", y=1.05), legend_title_text="",
        )
        fig_hr.update_xaxes(showgrid=False, dtick=2)
        fig_hr.update_yaxes(gridcolor="#252A38")
        # Annotations picos
        for _h, _lbl in [(8,"Rush AM"),(17,"Rush PM")]:
            fig_hr.add_vline(x=_h, line_dash="dot", line_color="#7A8499", line_width=1)
            fig_hr.add_annotation(x=_h, y=0, text=_lbl, showarrow=False,
                                  font=dict(size=9,color="#7A8499",family="IBM Plex Mono"),
                                  yref="paper", yanchor="bottom", xanchor="center")
        st.plotly_chart(fig_hr, width="stretch")

    # ── Fila 2: Día de semana + Condición climática ────────────────────────
    st.markdown("<div class='section-header' style='margin-top:20px'>Patrones por día de semana y condiciones climáticas</div>",
                unsafe_allow_html=True)
    c_dow, c_wx = st.columns(2, gap="medium")

    with c_dow:
        st.markdown("<div class='section-header'>Accidentes por día de la semana</div>",
                    unsafe_allow_html=True)
        _DOW_MAP = {0:"Lun",1:"Mar",2:"Mié",3:"Jue",4:"Vie",5:"Sáb",6:"Dom"}
        _dow_sev = fl.groupby(["DayOfWeek","MUTCD"]).size().reset_index(name="n")
        _dow_sev["Día"] = _dow_sev["DayOfWeek"].map(_DOW_MAP)
        fig_dow = px.bar(
            _dow_sev, x="Día", y="n", color="MUTCD",
            barmode="stack",
            category_orders={"Día":["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"],
                             "MUTCD":["Menor","Intermedio","Mayor"]},
            color_discrete_map={"Mayor":"#E74C3C","Intermedio":"#F39C12","Menor":"#3498DB"},
            template="plotly_dark",
            labels={"n":"Accidentes","Día":"Día","MUTCD":"Severidad"},
        )
        fig_dow.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
            margin=dict(l=0,r=0,t=10,b=0), height=280,
            legend=dict(orientation="h", y=1.05), legend_title_text="",
        )
        fig_dow.update_xaxes(showgrid=False)
        fig_dow.update_yaxes(gridcolor="#252A38")
        st.plotly_chart(fig_dow, width="stretch")

    with c_wx:
        st.markdown("<div class='section-header'>Top 10 condiciones climáticas — P(Mayor)</div>",
                    unsafe_allow_html=True)
        _wx_agg = (
            fl.groupby("Weather_Condition")
            .agg(total=("MUTCD","count"), mayor=("MUTCD", lambda x: (x=="Mayor").sum()))
            .query("total >= 50")
            .assign(p_mayor=lambda d: d["mayor"]/d["total"]*100)
            .sort_values("p_mayor", ascending=False)
            .head(10)
            .reset_index()
        )
        fig_wx = go.Figure(go.Bar(
            x=_wx_agg["p_mayor"],
            y=_wx_agg["Weather_Condition"],
            orientation="h",
            marker_color=[
                "#E74C3C" if p>=30 else "#F39C12" if p>=20 else "#3498DB"
                for p in _wx_agg["p_mayor"]
            ],
            text=[f"{p:.1f}%" for p in _wx_agg["p_mayor"]],
            textposition="outside",
            textfont=dict(size=10, color="#E8ECF4", family="IBM Plex Mono"),
        ))
        fig_wx.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
            margin=dict(l=0,r=55,t=10,b=10), height=280,
            xaxis=dict(title="P(Mayor) %", gridcolor="#252A38", tickfont=dict(color="#7A8499",size=10)),
            yaxis=dict(tickfont=dict(color="#E8ECF4",size=10)),
            showlegend=False, template="plotly_dark",
        )
        st.plotly_chart(fig_wx, width="stretch")

    # ── Fila 3: Features de infraestructura ──────────────────────────────
    st.markdown("<div class='section-header' style='margin-top:20px'>Infraestructura vial — impacto en severidad Mayor</div>",
                unsafe_allow_html=True)

    _INFRA_COLS = ["Junction","Traffic_Signal","Crossing","Roundabout",
                   "Station","Bump","Amenity","Stop","Railway"]
    _infra_rows = []
    for col in _INFRA_COLS:
        _sub = fl[fl[col] == True]
        if len(_sub) < 20:
            continue
        _p_mayor = (_sub["MUTCD"] == "Mayor").mean() * 100
        _p_base  = (fl["MUTCD"] == "Mayor").mean() * 100
        _infra_rows.append({"Feature": col, "P(Mayor) con feature": _p_mayor,
                            "P(Mayor) base": _p_base, "lift": _p_mayor / max(_p_base, 0.001)})
    _infra_df = pd.DataFrame(_infra_rows).sort_values("lift", ascending=True)

    fig_infra = go.Figure()
    fig_infra.add_trace(go.Bar(
        x=_infra_df["P(Mayor) base"],
        y=_infra_df["Feature"],
        orientation="h",
        name="Base (sin feature)",
        marker_color="#1F2D40",
    ))
    fig_infra.add_trace(go.Bar(
        x=_infra_df["P(Mayor) con feature"],
        y=_infra_df["Feature"],
        orientation="h",
        name="Con feature presente",
        marker_color=[
            "#E74C3C" if l>1.3 else "#F39C12" if l>1.1 else "#3498DB"
            for l in _infra_df["lift"]
        ],
        text=[f"lift {l:.2f}×" for l in _infra_df["lift"]],
        textposition="outside",
        textfont=dict(size=9, color="#E8ECF4", family="IBM Plex Mono"),
    ))
    fig_infra.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
        margin=dict(l=0,r=80,t=10,b=10), height=300,
        xaxis=dict(title="P(Mayor) %", gridcolor="#252A38", tickfont=dict(color="#7A8499",size=10)),
        yaxis=dict(tickfont=dict(color="#E8ECF4",size=10)),
        legend=dict(orientation="h", y=1.05, font=dict(size=10)),
        template="plotly_dark",
    )
    st.plotly_chart(fig_infra, width="stretch")

    # ── Callout insights → modelo ─────────────────────────────────────────
    st.markdown("""
    <div style='background:#13161D;border:1px solid rgba(41,128,185,0.4);border-radius:8px;
                padding:16px 20px;margin-top:8px'>
      <div style='font-family:IBM Plex Mono,monospace;font-size:9px;text-transform:uppercase;
                  letter-spacing:.12em;color:#2980B9;margin-bottom:10px'>
        🔗 Hallazgos FL → Features del Modelo
      </div>
      <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;font-size:12px;color:#E8ECF4;line-height:1.65'>
        <div>
          <div style='color:#3498DB;font-family:IBM Plex Mono,monospace;font-size:10px;
                      text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>⏰ Temporal</div>
          Dos picos claros a las <b>8h</b> y <b>17h</b> (rush hours) con mayor proporción de
          accidentes Mayores. El <b>DayOfWeek</b> y <b>Hour</b> se incluyeron directamente en el pipeline.
        </div>
        <div>
          <div style='color:#F39C12;font-family:IBM Plex Mono,monospace;font-size:10px;
                      text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>🌧 Clima</div>
          Niebla, tormenta eléctrica y lluvia fuerte elevan P(Mayor) >
          <b>25%</b>. Se mapearon a <b>Weather_Condition</b> y se añaden en tiempo real
          via Open-Meteo.
        </div>
        <div>
          <div style='color:#E74C3C;font-family:IBM Plex Mono,monospace;font-size:10px;
                      text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px'>🚦 Infraestructura</div>
          <b>Junction</b> y <b>Traffic_Signal</b> muestran el mayor lift de severidad.
          Ambas son toggleables en el sidebar del predictor para escenarios
          personalizados.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — ACTUARIAL
# ═══════════════════════════════════════════════════════════════════════════
with tab_actuarial:
    st.markdown("<div class='section-header'>Tabla actuarial por corregimiento — base para tarifación</div>", unsafe_allow_html=True)

    act_sorted = zones_dynamic.sort_values("prima_index_dyn", ascending=False).reset_index(drop=True)

    header_html = """
    <table class='prima-table'>
    <thead><tr>
      <th>#</th><th>Corregimiento</th><th>Nivel</th>
      <th>E[Freq]</th><th>YoY</th>
      <th>P(Mayor)</th><th>Índice Prima</th>
    </tr></thead><tbody>
    """
    rows_html = ""
    for i, row in act_sorted.iterrows():
        nivel = str(row["nivel_dyn"])
        pill_cls = {"Critico":"pill-critico","Alto":"pill-alto","Moderado":"pill-moderado","Bajo":"pill-bajo"}.get(nivel,"pill-bajo")
        yoy_col  = "#E74C3C" if row["YoY"] > 0 else "#2ECC71"
        pi_color = "#E74C3C" if row["prima_index_dyn"] >= 5 else ("#F39C12" if row["prima_index_dyn"] >= 2 else "#7A8499")
        rows_html += (
            "<tr>"
            "<td style='color:#7A8499;font-family:IBM Plex Mono,monospace;font-size:11px'>" + "{:02d}".format(i+1) + "</td>"
            "<td style='font-weight:500'>" + str(row["County"]) + "</td>"
            "<td><span class='risk-pill " + pill_cls + "'>" + nivel + "</span></td>"
            "<td style='font-family:IBM Plex Mono,monospace'>" + "{:.3f}".format(row["freq_pred_dyn"]) + "</td>"
            "<td style='color:" + yoy_col + ";font-family:IBM Plex Mono,monospace'>" + "{:+.1f}%".format(row["YoY"]*100) + "</td>"
            "<td style='font-family:IBM Plex Mono,monospace'>" + "{:.1f}%".format(row["p_mayor"]*100) + "</td>"
            "<td style='font-family:Syne,sans-serif;font-weight:700;font-size:1.1rem;color:" + pi_color + "'>" + "{:.2f}&times;".format(row["prima_index_dyn"]) + "</td>"
            "</tr>"
        )
    st.markdown(header_html + rows_html + "</tbody></table>", unsafe_allow_html=True)

    st.markdown("<div class='section-header' style='margin-top:32px'>Índice de prima técnica relativa (scatter)</div>", unsafe_allow_html=True)
    fig_act = px.scatter(act_sorted,
        x="freq_pred_dyn", y="p_mayor",
        size="prima_index_dyn", color="nivel_dyn",
        color_discrete_map={"Crítico":"#E74C3C","Alto":"#F39C12","Moderado":"#2ECC71","Bajo":"#3498DB"},
        text="County",
        labels={"freq_pred_dyn":"E[Frecuencia] Poisson","p_mayor":"P(Accidente Mayor)","nivel":"Nivel"},
        template="plotly_dark",
        size_max=50)
    fig_act.update_traces(textposition="top center", textfont_size=9, textfont_color="#7A8499")
    fig_act.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(19,22,29,1)",
        margin=dict(l=0,r=0,t=10,b=0), height=420,
        legend_title_text="Nivel de riesgo"
    )
    fig_act.update_xaxes(showgrid=True, gridcolor="#252A38")
    fig_act.update_yaxes(showgrid=True, gridcolor="#252A38")
    st.plotly_chart(fig_act, width='stretch')

    # ── Factores de Recargo por Vía ───────────────────────────────────────
    st.markdown("<div class='section-header' style='margin-top:32px'>Factores de Recargo (Micro-segmentación por Vía)</div>", unsafe_allow_html=True)
    
    _rt_act = df.groupby("Road_Type").apply(lambda x: pd.Series({
        "total": len(x),
        "p_mayor": (x["MUTCD_Category"]=="Mayor").mean() * 100
    }), include_groups=False).reset_index()
    _rt_act = _rt_act[_rt_act["total"] > 10]
    
    if "avenue" in _rt_act["Road_Type"].values:
        _base_p = _rt_act.loc[_rt_act["Road_Type"]=="avenue", "p_mayor"].values[0]
    else:
        _base_p = _rt_act["p_mayor"].min()
        
    _rt_act["recargo"] = _rt_act["p_mayor"] / max(_base_p, 0.001)
    
    # Aplicar un piso técnico (floor) actuarial del 0.75x para evitar recargos nulos (0.00x)
    _rt_act["recargo"] = _rt_act["recargo"].apply(lambda x: max(x, 0.75))
    _rt_act = _rt_act.sort_values("recargo", ascending=False)
    
    _cards_html = "<div style='display:flex;gap:12px;flex-wrap:wrap'>"
    for _, _row in _rt_act.iterrows():
        _rc_color = "#E74C3C" if _row["recargo"] > 1.5 else ("#F39C12" if _row["recargo"] > 1.1 else "#2ECC71")
        _p_text = "{:.2f}%".format(_row['p_mayor']) if _row['p_mayor'] > 0 else "&lt; 0.01%"
        _cards_html += (
            "<div style='flex:1;min-width:120px;text-align:center;padding:12px;"
            "background:#1A1E28;border-radius:6px;border:1px solid #252A38'>"
            "<div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:#E8ECF4;"
            "text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px'>"
            + str(_row['Road_Type'])
            + "</div><div style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:700;color:"
            + _rc_color + "'>"
            + "{:.2f}x".format(_row['recargo'])
            + "</div><div style='font-size:10px;color:#7A8499;margin-top:2px'>P(Mayor): "
            + _p_text
            + "</div></div>"
        )
    _cards_html += "</div>"
    st.markdown(_cards_html, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#13161D;border:1px solid #252A38;border-radius:8px;padding:16px 20px;margin-top:24px'>
      <div style='font-family:IBM Plex Mono,monospace;font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:#7A8499;margin-bottom:10px'>
        Metodología actuarial
      </div>
      <div style='font-size:13px;color:#E8ECF4;line-height:1.7'>
        <b>Prima técnica</b> = E[Frecuencia] × P(Mayor | siniestro) × E[Costo siniestro] × (1 + loading)<br>
        <b>Índice relativo</b> = (E[Freq_Poisson] × P(Mayor) × (1 + YoY)) / min(mismo producto)<br>
        <b>Calibración</b>: predicciones FL ajustadas con peso INEC por corregimiento<br>
        <span style='color:#7A8499'>Pendiente: dataset de montos reales para calibrar E[Costo]. Fuente sugerida: FEDPA con campos de liquidación.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

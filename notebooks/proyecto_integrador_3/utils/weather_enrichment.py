"""
Weather Enrichment — panama_synthetic_accidents.csv
====================================================
Fuente : Open-Meteo Historical Weather API  (ERA5 reanalysis, sin API key)
Docs   : https://open-meteo.com/en/docs/historical-weather-api
ERA5   : cobertura global desde 1940, resolución horaria ~25 km

Variables obtenidas por registro (fecha + hora + lat/lng):
  temperature_2m        → Temperature(F)
  relative_humidity_2m  → Humidity(%)
  precipitation         → Precipitation(in)
  weather_code          → Weather_Condition  (WMO code → string)
  wind_speed_10m        → Wind_Speed(mph)
  wind_gusts_10m        → Wind_Gusts(mph)    [nueva columna]
  cloud_cover           → Cloud_Cover(%)     [nueva columna]

Nota: `visibility` NO está disponible en el archivo ERA5 de Open-Meteo
(la API devuelve null). Se descarta la columna Visibility(mi) del CSV.

Estrategia para minimizar llamadas API
---------------------------------------
1. Coordenadas redondeadas a 2 decimales (~1 km) → colapsa puntos cercanos.
2. Agrupación por (date_str, lat_r, lng_r) → 1 llamada por punto-día.
3. Cache en disco (sqlite) con requests_cache → 0 llamadas en re-ejecuciones.
4. Checkpoint CSV → reanuda desde donde paró si el proceso se interrumpe.

Instalación de dependencias:
    pip install openmeteo-requests requests-cache retry-requests pandas tqdm

Uso CLI (desde cualquier directorio):
    python utils/weather_enrichment.py

    # O con rutas explícitas:
    python utils/weather_enrichment.py \\
        --input  notebooks/output/data/panama_synthetic_accidents.csv \\
        --output notebooks/output/data/panama_synthetic_accidents_weather.csv \\
        [--checkpoint utils/weather_checkpoint.csv] \\
        [--cache-dir  utils/.openmeteo_cache]

Nota: Los defaults de --input y --output apuntan a notebooks/output/data/
      relativo a la raíz del proyecto. El checkpoint y cache quedan en utils/.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

# ── Rutas por defecto relativas al script ──────────────────────────────────────────
# Las entradas/salidas de datos van a notebooks/output/data/
# El checkpoint y el cache se mantienen en utils/ (archivos temporales).
_SCRIPT_DIR    = Path(__file__).resolve().parent          # …/utils
_OUTPUT_DATA   = _SCRIPT_DIR.parent.parent / "output" / "data"

import numpy as np
import pandas as pd
import requests
import requests_cache
from retry_requests import retry
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── WMO Weather Code → legible string ────────────────────────────────────────
# Reference: https://open-meteo.com/en/docs#weathervariables
WMO_MAP: dict[int, str] = {
    0:  "Clear",
    1:  "Mostly Clear",
    2:  "Partly Cloudy",
    3:  "Overcast",
    45: "Fog",
    48: "Rime Fog",
    51: "Light Drizzle",
    53: "Drizzle",
    55: "Heavy Drizzle",
    56: "Light Freezing Drizzle",
    57: "Freezing Drizzle",
    61: "Light Rain",
    63: "Rain",
    65: "Heavy Rain",
    66: "Light Freezing Rain",
    67: "Freezing Rain",
    71: "Light Snow",
    73: "Snow",
    75: "Heavy Snow",
    77: "Snow Grains",
    80: "Rain Showers",
    81: "Rain Showers",
    82: "Heavy Rain Showers",
    85: "Snow Showers",
    86: "Heavy Snow Showers",
    95: "Thunderstorm",
    96: "Thunderstorm w/ Hail",
    99: "Thunderstorm w/ Heavy Hail",
}


def wmo_to_string(code) -> str:
    """Convierte un WMO weather code a string legible."""
    if pd.isna(code):
        return "Unknown"
    return WMO_MAP.get(int(code), f"WMO_{int(code)}")


# ── Constantes de API ─────────────────────────────────────────────────────────
_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Variables ERA5 disponibles en Open-Meteo Archive API
# IMPORTANTE: 'visibility' devuelve null → excluida.
_HOURLY_VARS = [
    "temperature_2m",       # °F
    "relative_humidity_2m", # %   (nombre correcto en la API v1)
    "precipitation",        # inch
    "weather_code",         # WMO (nombre correcto en la API v1)
    "wind_speed_10m",       # mph
    "wind_gusts_10m",       # mph — nuevo
    "cloud_cover",          # %   — nuevo
]


def _build_session(cache_dir: str = ".openmeteo_cache") -> requests.Session:
    """
    Crea una sesión HTTP con cache en disco (SQLite) y reintentos automáticos.
    El cache evita volver a llamar la API para puntos ya descargados.
    """
    cache_path = Path(cache_dir) / "cache"
    requests_cache.install_cache(
        str(cache_path),
        expire_after=7 * 24 * 3600,  # 7 días
        allowable_methods=["GET"],
    )
    return retry(
        requests_cache.CachedSession(),
        retries=5,
        backoff_factor=0.6,
    )


# ── Fetch weather para un punto-día ──────────────────────────────────────────
def _fetch_one_point(
    session: requests.Session,
    lat: float,
    lng: float,
    date_str: str,
    delay: float = 0.12,
) -> list[dict]:
    """
    Llama a Open-Meteo Archive para (lat, lng, date_str) y devuelve
    una lista de 24 dicts, uno por hora local (America/Panama, GMT-5).

    Si la llamada falla, devuelve 24 filas con NaN para todas las variables.
    """
    params = {
        "latitude":           lat,
        "longitude":          lng,
        "start_date":         date_str,
        "end_date":           date_str,
        "hourly":             ",".join(_HOURLY_VARS),
        "timezone":           "America/Panama",
        "temperature_unit":   "fahrenheit",
        "wind_speed_unit":    "mph",
        "precipitation_unit": "inch",
    }

    blank_row = {
        "lat_r": lat, "lng_r": lng, "date_str": date_str,
        "temp_f": np.nan, "humidity_pct": np.nan,
        "precip_in": np.nan, "wmo_code": np.nan,
        "wind_mph": np.nan, "gusts_mph": np.nan, "cloud_pct": np.nan,
    }

    try:
        resp = session.get(_API_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        n = len(hourly.get("time", [None] * 24))

        def _get(key):
            return hourly.get(key, [np.nan] * n)

        rows = []
        for h in range(n):
            rows.append({
                "lat_r":       lat,
                "lng_r":       lng,
                "date_str":    date_str,
                "hour":        h,
                "temp_f":      _get("temperature_2m")[h],
                "humidity_pct":_get("relative_humidity_2m")[h],
                "precip_in":   _get("precipitation")[h],
                "wmo_code":    _get("weather_code")[h],
                "wind_mph":    _get("wind_speed_10m")[h],
                "gusts_mph":   _get("wind_gusts_10m")[h],
                "cloud_pct":   _get("cloud_cover")[h],
            })

        # Rate-limit suave: solo si la respuesta vino de la red
        if not getattr(resp, "from_cache", False):
            time.sleep(delay)

        return rows

    except Exception as exc:
        log.warning("  ⚠ Error (%s, %s, %s): %s", lat, lng, date_str, exc)
        return [{**blank_row, "hour": h} for h in range(24)]


# ── Fetch batch con checkpoint ────────────────────────────────────────────────
def fetch_weather_for_points(
    unique_points: pd.DataFrame,
    session: requests.Session,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """
    Descarga datos horarios para cada (lat_r, lng_r, date_str) único.

    Parámetros
    ----------
    unique_points    : DataFrame con columnas [lat_r, lng_r, date_str]
    session          : sesión HTTP (con cache y retry)
    checkpoint_path  : ruta CSV para guardar progreso y reanudar si se corta

    Retorna
    -------
    DataFrame con columnas: lat_r, lng_r, date_str, hour, temp_f,
    humidity_pct, precip_in, wmo_code, wind_mph, gusts_mph, cloud_pct
    """
    # ── Carga checkpoint existente ────────────────────────────────────────
    done_keys: set[tuple] = set()
    previous_rows: list[dict] = []

    if checkpoint_path and Path(checkpoint_path).exists():
        prev = pd.read_csv(checkpoint_path)
        done_keys = set(zip(prev["lat_r"], prev["lng_r"], prev["date_str"]))
        previous_rows = prev.to_dict("records")
        log.info("  Checkpoint cargado: %d puntos ya procesados", len(done_keys) // 24)

    # ── Filtrar puntos pendientes ─────────────────────────────────────────
    mask_done = unique_points.apply(
        lambda r: (r["lat_r"], r["lng_r"], r["date_str"]) in done_keys, axis=1
    )
    pending = unique_points[~mask_done].reset_index(drop=True)
    log.info(
        "  Puntos únicos totales : %d | ya descargados : %d | pendientes : %d",
        len(unique_points), mask_done.sum(), len(pending),
    )
    log.info(
        "  Tiempo estimado (sin cache) : ~%.1f min",
        len(pending) * 0.15 / 60,
    )

    # ── Loop principal ────────────────────────────────────────────────────
    all_rows = list(previous_rows)
    SAVE_EVERY = 50   # guarda checkpoint cada N puntos procesados

    for idx, (_, row) in enumerate(
        tqdm(pending.iterrows(), total=len(pending), desc="Fetching ERA5"),
        start=1,
    ):
        rows = _fetch_one_point(
            session, row["lat_r"], row["lng_r"], row["date_str"]
        )
        all_rows.extend(rows)

        # Checkpoint periódico
        if checkpoint_path and idx % SAVE_EVERY == 0:
            pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
            log.debug("  Checkpoint guardado (%d/%d)", idx, len(pending))

    # Checkpoint final
    if checkpoint_path:
        pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
        log.info("  Checkpoint final guardado → %s", checkpoint_path)

    return pd.DataFrame(all_rows)


# ── Pipeline principal ────────────────────────────────────────────────────────
def enrich_weather(
    input_csv: str,
    output_csv: str,
    checkpoint_path: str | None = None,
    cache_dir: str = ".openmeteo_cache",
) -> pd.DataFrame:
    """
    Lee el CSV sintético de Panamá, enriquece cada registro con datos
    climáticos reales de ERA5 (Open-Meteo Archive) y guarda el resultado.

    Columnas REEMPLAZADAS con datos observados (no sintéticos):
        Temperature(F), Humidity(%), Precipitation(in),
        Wind_Speed(mph), Weather_Condition

    Columnas NUEVAS añadidas:
        Wind_Gusts(mph), Cloud_Cover(%)

    Columnas ELIMINADAS (no disponibles en ERA5 via Open-Meteo):
        Visibility(mi)  ← ERA5 no provee visibilidad horaria
    """
    log.info("═" * 60)
    log.info("Cargando %s …", input_csv)
    df = pd.read_csv(input_csv)
    log.info("  Shape original: %s", df.shape)

    # ── Preparar claves de join ───────────────────────────────────────────
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    df["date_str"]   = df["Start_Time"].dt.strftime("%Y-%m-%d")
    df["hour_int"]   = df["Hour"].astype(int)

    # Redondear a 2 dec (~1 km) → colapsa puntos del mismo corregimiento
    df["lat_r"] = df["Start_Lat"].round(2)
    df["lng_r"] = df["Start_Lng"].round(2)

    # ── Puntos únicos ─────────────────────────────────────────────────────
    unique_pts = (
        df[["lat_r", "lng_r", "date_str"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    log.info("  Puntos únicos (lat×lng×día): %d", len(unique_pts))

    # ── Sesión HTTP ───────────────────────────────────────────────────────
    session = _build_session(cache_dir)

    # ── Descarga ERA5 ─────────────────────────────────────────────────────
    weather_df = fetch_weather_for_points(unique_pts, session, checkpoint_path)

    # ── Mapear WMO code → string legible ──────────────────────────────────
    weather_df["weather_str"] = weather_df["wmo_code"].apply(wmo_to_string)

    # ── Join al dataset original ──────────────────────────────────────────
    df_enriched = df.merge(
        weather_df.rename(columns={"hour": "hour_int"}),
        on=["lat_r", "lng_r", "date_str", "hour_int"],
        how="left",
    )

    # ── Reemplazar variables sintéticas con observaciones reales ──────────
    mask = df_enriched["temp_f"].notna()
    log.info(
        "\n  Registros con ERA5 (%% cobertura): %d / %d  (%.1f%%)",
        mask.sum(), len(df_enriched), mask.mean() * 100,
    )

    df_enriched.loc[mask, "Temperature(F)"]    = df_enriched.loc[mask, "temp_f"].round(1)
    df_enriched.loc[mask, "Humidity(%)"]       = df_enriched.loc[mask, "humidity_pct"].round(1)
    df_enriched.loc[mask, "Precipitation(in)"] = df_enriched.loc[mask, "precip_in"].round(4)
    df_enriched.loc[mask, "Wind_Speed(mph)"]   = df_enriched.loc[mask, "wind_mph"].round(1)
    df_enriched.loc[mask, "Weather_Condition"] = df_enriched.loc[mask, "weather_str"]

    # Columnas nuevas con datos ERA5 (siempre presentes, NaN donde API falló)
    df_enriched["Wind_Gusts(mph)"]  = df_enriched["gusts_mph"].round(1)
    df_enriched["Cloud_Cover(%)"]   = df_enriched["cloud_pct"].round(0).astype("Int64")

    # ── Eliminar Visibility(mi) — ERA5 no la provee ───────────────────────
    if "Visibility(mi)" in df_enriched.columns:
        df_enriched = df_enriched.drop(columns=["Visibility(mi)"])
        log.info("  Columna Visibility(mi) eliminada (no disponible en ERA5).")

    # ── Recalcular Sunrise_Sunset con umbral solar de Panamá ──────────────
    # Amanecer ≈ 06:00, Atardecer ≈ 18:30 → usamos 6-18 (h local)
    df_enriched["Sunrise_Sunset"] = df_enriched["Hour"].apply(
        lambda h: "Day" if 6 <= int(h) <= 18 else "Night"
    )

    # ── Drop columnas auxiliares ──────────────────────────────────────────
    aux_cols = [
        "date_str", "hour_int", "lat_r", "lng_r",
        "temp_f", "humidity_pct", "precip_in",
        "wmo_code", "wind_mph", "gusts_mph", "cloud_pct", "weather_str",
    ]
    df_enriched = df_enriched.drop(
        columns=[c for c in aux_cols if c in df_enriched.columns]
    )

    # ── QA ────────────────────────────────────────────────────────────────
    log.info("\n═══ QA post-enriquecimiento ═══")
    log.info("Shape final : %s", df_enriched.shape)
    log.info("Nulos totales: %d", df_enriched.isnull().sum().sum())

    log.info("\nDistribución Weather_Condition (ERA5):")
    log.info("\n%s", df_enriched["Weather_Condition"].value_counts().head(10).to_string())

    log.info("\nTemperature(F):")
    log.info("\n%s", df_enriched["Temperature(F)"].describe().round(2).to_string())

    log.info("\nPrecipitation(in):")
    log.info("\n%s", df_enriched["Precipitation(in)"].describe().round(5).to_string())

    log.info("\nCloud_Cover(%%):")
    log.info("\n%s", df_enriched["Cloud_Cover(%)"].describe().round(1).to_string())

    # ── Guardar ───────────────────────────────────────────────────────────
    df_enriched.to_csv(output_csv, index=False)
    log.info("\n✅ CSV enriquecido guardado → %s", output_csv)
    log.info("═" * 60)

    return df_enriched


# ── Live weather (Forecast API) ──────────────────────────────────────────────
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Variables disponibles en el Forecast API (same names as Archive)
_FORECAST_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "visibility",          # sí disponible en Forecast (a diferencia de ERA5)
]


def get_live_weather(
    lat: float,
    lng: float,
    target_dt,           # datetime object, local Panama time
    climatology: dict | None = None,
    cache_dir: str = ".openmeteo_cache",
) -> dict:
    """
    Devuelve un dict con condiciones climáticas para (lat, lng, datetime).

    Estrategia:
      1. Si la fecha está dentro de los próximos 16 días o los últimos 5 días
         → Open-Meteo Forecast API (datos reales / pronóstico).
      2. En caso contrario → climatología precalculada del CSV ERA5.
         Si no se pasa climatología → valores típicos de Panama City.

    Retorna
    -------
    dict con claves:
        temperature_f, humidity_pct, precip_in, wind_mph, gusts_mph,
        cloud_pct, visibility_mi, weather_condition, weather_code,
        source  ("forecast" | "climatology")
    """
    import datetime as _dt

    if isinstance(target_dt, _dt.date) and not isinstance(target_dt, _dt.datetime):
        target_dt = _dt.datetime(target_dt.year, target_dt.month, target_dt.day, 12)

    now       = _dt.datetime.now()
    delta     = (target_dt.date() - now.date()).days   # días desde hoy

    # ── Intento Forecast API (−60 … +16 días) ────────────────────────────────
    # Open-Meteo Forecast soporta hasta past_days=92 (pasado) y +16 futuro.
    if -60 <= delta <= 16:
        try:
            sess = _build_session(cache_dir)
            params = {
                "latitude":           round(lat, 4),
                "longitude":          round(lng, 4),
                "start_date":         target_dt.strftime("%Y-%m-%d"),
                "end_date":           target_dt.strftime("%Y-%m-%d"),
                "hourly":             ",".join(_FORECAST_VARS),
                "timezone":           "America/Panama",
                "temperature_unit":   "fahrenheit",
                "wind_speed_unit":    "mph",
                "precipitation_unit": "inch",
                # past_days soportado hasta 92; para fechas futuras se ignora.
                "past_days":          max(0, -delta),
            }
            resp = sess.get(_FORECAST_URL, params=params, timeout=10)
            resp.raise_for_status()
            data   = resp.json()
            hourly = data.get("hourly", {})

            h = target_dt.hour
            def _v(key, default=None):
                arr = hourly.get(key, [])
                return arr[h] if h < len(arr) else default

            wmo  = _v("weather_code")
            vis_m = _v("visibility")          # metros en Forecast
            vis_mi = (vis_m / 1609.34) if vis_m is not None else 7.0
            vis_mi = round(min(vis_mi, 10.0), 2)

            return {
                "temperature_f":     _v("temperature_2m",      84.0),
                "humidity_pct":      _v("relative_humidity_2m", 80.0),
                "precip_in":         _v("precipitation",         0.0),
                "wind_mph":          _v("wind_speed_10m",        8.0),
                "gusts_mph":         _v("wind_gusts_10m",       12.0),
                "cloud_pct":         _v("cloud_cover",          50.0),
                "visibility_mi":     vis_mi,
                "weather_code":      wmo,
                "weather_condition": wmo_to_string(wmo),
                "source":            "forecast",
            }
        except Exception as exc:
            log.warning("Forecast API failed (%s); falling back to climatology. %s", target_dt, exc)

    # ── Fallback: climatología ─────────────────────────────────────────────
    if climatology:
        key = (round(lat, 2), round(lng, 2), target_dt.month, target_dt.hour)
        # Try exact key, then month+hour only, then just month
        row = (
            climatology.get(key)
            or climatology.get((None, None, target_dt.month, target_dt.hour))
            or climatology.get((None, None, target_dt.month, None))
        )
        if row:
            return {**row, "source": "climatology"}

    # ── Defaults de Panama City si todo falla ─────────────────────────────
    log.warning("No climatology available; using Panama City defaults.")
    is_wet = target_dt.month in {5, 6, 7, 8, 9, 10, 11}
    return {
        "temperature_f":     82.0 if is_wet else 88.0,
        "humidity_pct":      88.0 if is_wet else 72.0,
        "precip_in":          0.05 if is_wet else 0.0,
        "wind_mph":            8.0,
        "gusts_mph":          14.0,
        "cloud_pct":          80.0 if is_wet else 40.0,
        "visibility_mi":       7.0,
        "weather_code":        None,
        "weather_condition": "Rain" if is_wet else "Clear",
        "source":            "default",
    }


def build_climatology(enriched_csv: str) -> dict:
    """
    Pre-computa medianas climáticas por (lat_r, lng_r, Month, Hour)
    desde el CSV enriquecido con ERA5.

    Retorna un dict  {(lat_r, lng_r, month, hour): {...valores...}}
    listo para pasarse a get_live_weather(climatology=...).

    También guarda claves (None, None, month, hour) con la mediana
    global por mes+hora (fallback de segundo nivel).
    """
    df = pd.read_csv(enriched_csv)
    df["lat_r"] = df["Start_Lat"].round(2)
    df["lng_r"] = df["Start_Lng"].round(2)

    num_cols = {
        "Temperature(F)":    "temperature_f",
        "Humidity(%)":       "humidity_pct",
        "Precipitation(in)": "precip_in",
        "Wind_Speed(mph)":   "wind_mph",
        "Wind_Gusts(mph)":   "gusts_mph",
        "Cloud_Cover(%)":    "cloud_pct",
    }

    result: dict = {}

    # ── Nivel 1: por zona×mes×hora ─────────────────────────────────────
    grp = df.groupby(["lat_r", "lng_r", "Month", "Hour"])
    for (lat_r, lng_r, month, hour), g in grp:
        entry: dict = {}
        for csv_col, key in num_cols.items():
            if csv_col in g.columns:
                entry[key] = round(float(g[csv_col].median()), 2)
        # Visibility no está en ERA5 → usar 7.0 como proxy Panama City
        entry["visibility_mi"] = 7.0
        # Moda de Weather_Condition
        if "Weather_Condition" in g.columns:
            mode_series = g["Weather_Condition"].mode()
            entry["weather_condition"] = mode_series.iloc[0] if len(mode_series) else "Clear"
        entry["weather_code"] = None
        result[(lat_r, lng_r, month, hour)] = entry

    # ── Nivel 2: global mes×hora ───────────────────────────────────────
    grp2 = df.groupby(["Month", "Hour"])
    for (month, hour), g in grp2:
        entry = {}
        for csv_col, key in num_cols.items():
            if csv_col in g.columns:
                entry[key] = round(float(g[csv_col].median()), 2)
        entry["visibility_mi"] = 7.0
        if "Weather_Condition" in g.columns:
            mode_series = g["Weather_Condition"].mode()
            entry["weather_condition"] = mode_series.iloc[0] if len(mode_series) else "Clear"
        entry["weather_code"] = None
        result[(None, None, month, hour)] = entry

    log.info("Climatology built: %d zona-entries + %d global-entries",
             len([k for k in result if k[0] is not None]),
             len([k for k in result if k[0] is None]))
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enriquece panama_synthetic_accidents.csv con ERA5 "
                    "via Open-Meteo Archive API."
    )
    p.add_argument(
        "--input", "-i",
        default=str(_OUTPUT_DATA / "panama_synthetic_accidents.csv"),
        help="Ruta al CSV de entrada (default: %(default)s)",
    )
    p.add_argument(
        "--output", "-o",
        default=str(_OUTPUT_DATA / "panama_synthetic_accidents_weather.csv"),
        help="Ruta del CSV de salida (default: %(default)s)",
    )
    p.add_argument(
        "--checkpoint", "-c",
        default=str(_SCRIPT_DIR / "weather_checkpoint.csv"),
        help="CSV de checkpoint para reanudar si se interrumpe (default: %(default)s)",
    )
    p.add_argument(
        "--cache-dir",
        default=str(_SCRIPT_DIR / ".openmeteo_cache"),
        help="Directorio para el cache HTTP (default: %(default)s)",
    )
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Desactiva el sistema de checkpoint",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    ckpt = None if args.no_checkpoint else args.checkpoint

    df_final = enrich_weather(
        input_csv       = args.input,
        output_csv      = args.output,
        checkpoint_path = ckpt,
        cache_dir       = args.cache_dir,
    )

    print("\n── Primeras 3 filas — columnas climáticas ──")
    climate_cols = [
        c for c in [
            "County", "Hour", "DayOfWeek",
            "Temperature(F)", "Humidity(%)", "Precipitation(in)",
            "Wind_Speed(mph)", "Wind_Gusts(mph)", "Cloud_Cover(%)",
            "Weather_Condition", "Sunrise_Sunset", "Target_Severity",
        ] if c in df_final.columns
    ]
    print(df_final[climate_cols].head(3).to_string(index=False))


"""
streaming_job.py — Spark Streaming + predicción MLlib
======================================================
Recibe datos del backend vía socket TCP (puerto 9999).
Cada mensaje es un JSON con:
  timestamp, vehicle_count, estado, hour, weekday, minute, is_rush_hour,
  tramo1_brightness_mean, tramo1_brightness_std, tramo1_saturation_mean,
  tramo1_edge_density, tramo1_gray_contrast,
  tramo2_*, tramo3_*

Por cada micro-batch (30s) calcula:
  - Media móvil de vehículos en ventana de 5 minutos
  - Tendencia (SUBIENDO / BAJANDO / ESTABLE)
  - Predicción MLlib del estado para ese instante (si el modelo está disponible)
  - Guarda el agregado en PostgreSQL (tabla spark_ventanas)

Nota: el modelo se carga una vez al arrancar. Si se reentrena
(train_spark.py), reiniciar este servicio para recoger el nuevo modelo:
  docker compose restart spark
"""

import os
import json
import numpy as np
import psycopg2
import joblib

from pathlib import Path
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from typing import Optional

# ─── Config ───────────────────────────────────────────────────
DATABASE_URL   = os.environ.get("DATABASE_URL", "postgresql://rotonda:rotonda123@postgres:5432/rotonda")
MODELS_DIR     = Path("/data/models")
BATCH_INTERVAL = 30   # segundos por micro-batch

# ─── Features (mismo orden que train_spark.py) ────────────────
META_COLS = ["hour", "weekday", "minute", "is_rush_hour"]
VIS_COLS  = [
    f"{roi}_{feat}"
    for roi in ["tramo1_rotonda", "tramo2_rotonda", "tramo3_rotonda"]
    for feat in ["brightness_mean", "brightness_std",
                 "saturation_mean", "edge_density", "gray_contrast"]
]
FEATURE_COLS = META_COLS + VIS_COLS   # 19 features


# ═══════════════════════════════════════════════════════════════
#  Carga del modelo al arrancar (una sola vez en el driver)
# ═══════════════════════════════════════════════════════════════

def _load_model():
    model_path   = MODELS_DIR / "traffic_model.joblib"
    encoder_path = MODELS_DIR / "label_encoder.joblib"
    names_path   = MODELS_DIR / "feature_names.json"

    if not model_path.exists():
        print("[Spark] ⚠️  Modelo no encontrado — predicción MLlib desactivada")
        print(f"[Spark]    Ejecuta train_spark.py para entrenar el modelo")
        return None, None, None

    try:
        model        = joblib.load(model_path)
        encoder      = joblib.load(encoder_path)
        feature_names = json.loads(names_path.read_text())
        print(f"[Spark] ✅ Modelo cargado desde {MODELS_DIR}")
        print(f"[Spark]    Clases: {list(encoder.classes_)}")
        print(f"[Spark]    Features: {len(feature_names)}")
        return model, encoder, feature_names
    except Exception as e:
        print(f"[Spark] ⚠️  Error cargando modelo: {e}")
        return None, None, None


MODEL, ENCODER, FEATURE_NAMES = _load_model()


def _predict_estado(data: dict) -> Optional[str]:
    """
    Usa el modelo sklearn (exportado por train_spark.py) para predecir
    el estado a partir de las features del mensaje de streaming.
    Devuelve None si el modelo no está disponible o faltan features.
    """
    if MODEL is None:
        return None
    try:
        vec = np.array(
            [[float(data.get(f, 0.0)) for f in FEATURE_NAMES]],
            dtype=np.float32,
        )
        idx   = MODEL.predict(vec)[0]
        label = ENCODER.inverse_transform([idx])[0]
        return label
    except Exception as e:
        print(f"[Spark] ⚠️  Error en predicción: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  Spark Streaming
# ═══════════════════════════════════════════════════════════════

sc  = SparkContext(appName="RotondaStreaming")
ssc = StreamingContext(sc, BATCH_INTERVAL)
sc.setLogLevel("WARN")

lines = ssc.socketTextStream('backend', 9999)
records = (
    lines
    .map(lambda line: _safe_parse(line))
    .filter(lambda x: x is not None)
)


def _safe_parse(line: str):
    try:
        return json.loads(line.strip())
    except Exception:
        return None


# Ventana deslizante de 5 minutos
windowed = records.window(windowDuration=300, slideDuration=BATCH_INTERVAL)


def process_window(rdd):
    if rdd.isEmpty():
        return

    data   = rdd.collect()
    counts = [d["vehicle_count"] for d in data]

    avg_count  = sum(counts) / len(counts)
    max_count  = max(counts)

    # Tendencia
    if len(counts) >= 3:
        recent = sum(counts[-3:]) / 3
        older  = sum(counts[:3])  / 3
        if   recent > older * 1.2: tendencia = "SUBIENDO"
        elif recent < older * 0.8: tendencia = "BAJANDO"
        else:                      tendencia = "ESTABLE"
    else:
        tendencia = "ESTABLE"

    # Estado más frecuente en la ventana (ground truth del backend)
    estados = [d["estado"] for d in data]
    estado_ventana = max(set(estados), key=estados.count)

    # Predicción MLlib con el último sample de la ventana
    ultimo       = data[-1]
    estado_pred  = _predict_estado(ultimo)
    prediccion = estado_ventana  # usar el estado clasificado por el backend

    # Log
    pred_tag = f"pred={prediccion}" if estado_pred else "pred=N/A (sin modelo)"
    print(
        f"[Spark] ventana 5min | "
        f"avg={avg_count:.1f}  max={max_count}  "
        f"tendencia={tendencia}  estado={estado_ventana}  {pred_tag}"
    )

    # Guardar en PostgreSQL
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute(
            """
            INSERT INTO spark_ventanas
                (avg_vehicles, max_vehicles, tendencia,
                estado_ventana, estado_predicho, n_samples)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (avg_count, max_count, tendencia,
            estado_ventana, prediccion, len(counts)),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[Spark] Error PostgreSQL: {e}")


windowed.foreachRDD(process_window)

ssc.start()
ssc.awaitTermination()
"""
classifier.py — Clasificador de estado con RandomForest entrenado.
Requiere que existan en models/:
  - traffic_model.joblib
  - label_encoder.joblib
  - feature_names.json

Si el modelo no está disponible, cae de vuelta a umbrales fijos.
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))

# Umbrales de fallback
UMBRAL_DENSO   = 5
UMBRAL_COLAPSO = 10

# Score histórico de congestión por hora (calculado sobre el dataset completo)
# Horas fuera del rango observado usan 0.15 como valor neutro
CONGESTION_SCORE_BY_HOUR = {
    8:  0.203,
    9:  0.163,
    10: 0.238,
    11: 0.322,
    12: 0.437,
    13: 0.460,
    14: 0.365,
    15: 0.433,
    16: 0.420,
    17: 0.219,
    18: 0.098,
    19: 0.056,
}
PEAK_HOUR_START = 11
PEAK_HOUR_END   = 16

_model         = None
_encoder       = None
_feature_names = None


def _load_model():
    global _model, _encoder, _feature_names
    try:
        _model         = joblib.load(MODELS_DIR / "traffic_model.joblib")
        _encoder       = joblib.load(MODELS_DIR / "label_encoder.joblib")
        _feature_names = json.loads((MODELS_DIR / "feature_names.json").read_text())
        print(f"[Classifier] Modelo cargado desde {MODELS_DIR}")
    except Exception as e:
        print(f"[Classifier] Modelo no disponible ({e}) — usando umbrales fijos")
        _model = None


_load_model()


def predict(vehicle_count: int, hour: int, is_rush_hour: bool,
            visual_features: dict = None) -> dict:
    """
    visual_features: dict con las features visuales extraídas por detector.py.
    Si es None o el modelo no está cargado, usa umbrales fijos.
    """
    if _model is not None and visual_features is not None:
        try:
            now = datetime.now()
            meta = {
                "hour":                  hour,
                "weekday":               now.weekday(),
                "minute":                now.minute,
                "is_rush_hour":          int(is_rush_hour),
                "vehicle_count":         vehicle_count,
                "congestion_hour_score": CONGESTION_SCORE_BY_HOUR.get(hour, 0.15),
                "is_peak_hour":          1 if PEAK_HOUR_START <= hour <= PEAK_HOUR_END else 0,
            }
            all_features = {**meta, **visual_features}
            x = np.array([[all_features[f] for f in _feature_names]])

            pred_enc   = _model.predict(x)[0]
            pred_proba = _model.predict_proba(x)[0]
            estado     = _encoder.inverse_transform([pred_enc])[0]
            confianza  = float(pred_proba.max())

            return {"estado": estado, "confianza": confianza, "metodo": "randomforest"}
        except Exception as e:
            print(f"[Classifier] Error en predicción ML ({e}), usando umbrales")

    # Fallback: umbrales fijos
    if vehicle_count >= UMBRAL_COLAPSO:
        estado = "COLAPSO"
    elif vehicle_count >= UMBRAL_DENSO:
        estado = "DENSO"
    else:
        estado = "FLUIDO"
    return {"estado": estado, "confianza": 1.0, "metodo": "umbral"}
"""
train_spark.py — Entrenamiento con Spark MLlib
===============================================
Ejecutar manualmente cuando quieras reentrenar el modelo:

  docker exec rotonda_spark \
    /opt/spark/bin/spark-submit \
      --master local[*] \
      --driver-memory 6g \
      --conf spark.executor.memory=6g \
      --conf spark.memory.fraction=0.8 \
      --conf spark.sql.shuffle.partitions=4 \
      /opt/spark_jobs/train_spark.py \
    2>/dev/null | tee /tmp/train_output.txt

El script:
  1. Lee todos los labels.csv del dataset montado en /data/dataset
  2. Extrae las 15 features visuales con OpenCV vía mapPartitions (distribuido)
  3. Incluye vehicle_count del CSV (calculado por enrich_vehicle_count.py)
  4. Añade congestion_hour_score e is_peak_hour (franjas pico históricas)
  5. Balancea clases con sample_weight (inversamente proporcional a frecuencia)
  6. Entrena un Pipeline MLlib: StringIndexer → VectorAssembler → RandomForest
  7. Valida con CrossValidator 5-fold
  8. Exporta traffic_model.joblib + label_encoder.joblib → /data/models
"""

import json
import warnings
import joblib
import numpy as np
import cv2

from pathlib import Path
from typing import Iterator, Optional

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType,
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─── Rutas dentro del contenedor ─────────────────────────────
DATASET_DIR = Path("/data/dataset")
MODELS_DIR  = Path("/data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Geometría de la cámara (máscara interior rotonda) ────────
TARGET_W, TARGET_H = 640, 480

MASK_POLYGON = np.array([
    (638, 304), (539, 274), (495, 262), (459, 249),
    (437, 245), (411, 244), (386, 239), (365, 228),
    (349, 226), (323, 222), (299, 221), (277, 221),
    (245, 222), (227, 222), (209, 216), (194, 210),
    (168, 204), (150, 204), (136, 203), (121, 197),
    (106, 196), (83,  192), (68,  199), (37,  203),
    (17,  206), (7,   209), (5,   275), (94,  274),
    (141, 289), (209, 305), (272, 313), (331, 327),
    (368, 346), (418, 361), (473, 381), (508, 389),
    (536, 396), (577, 410), (601, 411), (624, 419),
    (633, 420), (639, 423),
], dtype=np.int32)

ROIS = {
    "tramo1_rotonda": (413, 246, 640, 404),
    "tramo2_rotonda": (184, 200, 409, 337),
    "tramo3_rotonda": (3,   158, 179, 296),
}

LABEL_ORDER = ["FLUIDO", "DENSO", "COLAPSO"]

# ─── Score histórico de congestión por hora ───────────────────
# Calculado del dataset completo: % frames DENSO+COLAPSO por hora.
# Horas fuera del rango observado usan 0.15 como valor neutro.
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

# ─── Features ────────────────────────────────────────────────
META_COLS = [
    "hour", "weekday", "minute", "is_rush_hour",
    "vehicle_count",
    "congestion_hour_score",   # % histórico de congestión para esa hora
    "is_peak_hour",            # 1 si 11 <= hour <= 16
]
VIS_COLS = [
    f"{roi}_{feat}"
    for roi in ROIS
    for feat in ["brightness_mean", "brightness_std",
                 "saturation_mean", "edge_density", "gray_contrast"]
]
FEATURE_COLS = META_COLS + VIS_COLS   # 22 en total


# ═══════════════════════════════════════════════════════════════
#  Extracción de features visuales (se ejecuta en cada worker)
# ═══════════════════════════════════════════════════════════════

def _letterbox(img):
    h, w    = img.shape[:2]
    scale   = min(TARGET_W / w, TARGET_H / h)
    nw, nh  = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((TARGET_H, TARGET_W, 3), 114, dtype=np.uint8)
    px, py  = (TARGET_W - nw) // 2, (TARGET_H - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, px, py, scale, w, h


def _extract_features(img_path: str) -> Optional[dict]:
    img = cv2.imread(img_path)
    if img is None:
        return None

    canvas, px, py, scale, ow, oh = _letterbox(img)

    road_mask = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
    poly_lb   = MASK_POLYGON.astype(np.float32).copy()
    poly_lb[:, 0] = poly_lb[:, 0] * (ow / TARGET_W) * scale + px
    poly_lb[:, 1] = poly_lb[:, 1] * (oh / TARGET_H) * scale + py
    cv2.fillPoly(
        road_mask,
        [np.clip(poly_lb, 0, [TARGET_W - 1, TARGET_H - 1]).astype(np.int32)],
        255,
    )

    hsv   = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray  = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = cv2.Canny(
        cv2.GaussianBlur(gray.astype(np.uint8), (3, 3), 0), 50, 150
    )

    features = {}
    for name, (x1, y1, x2, y2) in ROIS.items():
        ax1 = max(0,            int(x1 * (ow / TARGET_W) * scale + px))
        ay1 = max(0,            int(y1 * (oh / TARGET_H) * scale + py))
        ax2 = min(TARGET_W - 1, int(x2 * (ow / TARGET_W) * scale + px))
        ay2 = min(TARGET_H - 1, int(y2 * (oh / TARGET_H) * scale + py))

        roi_mask = road_mask[ay1:ay2, ax1:ax2]
        px_count = roi_mask.sum() / 255 + 1
        v_roi = hsv[ay1:ay2, ax1:ax2, 2]
        s_roi = hsv[ay1:ay2, ax1:ax2, 1]
        g_roi = gray[ay1:ay2, ax1:ax2]
        e_roi = edges[ay1:ay2, ax1:ax2]
        m     = roi_mask / 255

        features[f"{name}_brightness_mean"] = float(np.sum(v_roi * m) / px_count)
        features[f"{name}_brightness_std"]  = float(v_roi[roi_mask > 0].std() if roi_mask.any() else 0)
        features[f"{name}_saturation_mean"] = float(np.sum(s_roi * m) / px_count)
        features[f"{name}_edge_density"]    = float(np.sum(e_roi * m) / px_count / 255)
        features[f"{name}_gray_contrast"]   = float(g_roi[roi_mask > 0].std() if roi_mask.any() else 0)
    return features


def process_partition(rows: Iterator[Row]) -> Iterator[Row]:
    """Se ejecuta en cada worker — OpenCV por partición."""
    for row in rows:
        vis = _extract_features(row["img_path"])
        if vis is None:
            continue
        yield Row(
            hour                  = row["hour"],
            weekday               = row["weekday"],
            minute                = row["minute"],
            is_rush_hour          = row["is_rush_hour"],
            vehicle_count         = row["vehicle_count"],
            congestion_hour_score = row["congestion_hour_score"],
            is_peak_hour          = row["is_peak_hour"],
            estado                = row["estado"],
            fecha                 = row["fecha"],
            **{k: float(v) for k, v in vis.items()},
        )


def _feature_schema() -> StructType:
    fields = [
        StructField("hour",                  IntegerType(), True),
        StructField("weekday",               IntegerType(), True),
        StructField("minute",                IntegerType(), True),
        StructField("is_rush_hour",          IntegerType(), True),
        StructField("vehicle_count",         IntegerType(), True),
        StructField("congestion_hour_score", DoubleType(),  True),
        StructField("is_peak_hour",          IntegerType(), True),
        StructField("estado",                StringType(),  True),
        StructField("fecha",                 StringType(),  True),
    ]
    for col in VIS_COLS:
        fields.append(StructField(col, DoubleType(), True))
    return StructType(fields)


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

def _safe_int(value, default=0) -> int:
    """Convierte a int ignorando NaN, None y strings vacíos."""
    try:
        f = float(value)
        return default if f != f else int(f)  # f != f es True solo para NaN
    except (TypeError, ValueError):
        return default


def add_sample_weights(df, label_col="estado"):
    """
    Añade columna sample_weight con peso inversamente proporcional
    a la frecuencia de cada clase:
        peso = total / (n_clases × frecuencia_clase)
    """
    counts = df.groupBy(label_col).count().collect()
    total  = df.count()
    n_cls  = len(counts)

    weight_map = {row[label_col]: total / (n_cls * row["count"]) for row in counts}

    print(f"\n⚖️  Pesos por clase (total={total}, n_clases={n_cls}):")
    for estado, peso in sorted(weight_map.items()):
        print(f"     {estado:10s} → {peso:.4f}")

    expr = None
    for estado, peso in weight_map.items():
        cond = F.when(F.col(label_col) == estado, F.lit(float(peso)))
        expr = cond if expr is None else expr.when(F.col(label_col) == estado, F.lit(float(peso)))
    expr = expr.otherwise(F.lit(1.0))

    return df.withColumn("sample_weight", expr)


# ═══════════════════════════════════════════════════════════════
#  Catálogo: lee todos los labels.csv
# ═══════════════════════════════════════════════════════════════

def build_catalog(spark: SparkSession):
    import pandas as pd

    rows = []
    for day_dir in sorted(d for d in DATASET_DIR.iterdir() if d.is_dir()):
        csv_path   = day_dir / "labels.csv"
        frames_dir = day_dir / "frames"
        if not csv_path.exists():
            print(f"  ⚠️  Sin labels.csv en {day_dir.name}")
            continue
        labels = pd.read_csv(csv_path)
        has_vc = "vehicle_count" in labels.columns
        if not has_vc:
            print(f"  ⚠️  {day_dir.name} sin vehicle_count — usando 0")
        print(f"  📅 {day_dir.name}: {len(labels)} etiquetas")
        for _, row in labels.iterrows():
            img_path = frames_dir / row["filename"]
            if not img_path.exists():
                continue
            hour = _safe_int(row.get("hour"), 0)
            rows.append({
                "img_path":             str(img_path),
                "hour":                 hour,
                "weekday":              _safe_int(row.get("weekday"),      0),
                "minute":               _safe_int(row.get("minute"),       0),
                "is_rush_hour":         _safe_int(row.get("is_rush_hour"), 0),
                "vehicle_count":        _safe_int(row["vehicle_count"],    0) if has_vc else 0,
                "congestion_hour_score": CONGESTION_SCORE_BY_HOUR.get(hour, 0.15),
                "is_peak_hour":         1 if PEAK_HOUR_START <= hour <= PEAK_HOUR_END else 0,
                "estado":               row["estado"],
                "fecha":                day_dir.name,
            })

    print(f"\n  ✅ Catálogo: {len(rows)} imágenes")
    schema = StructType([
        StructField("img_path",              StringType(),  False),
        StructField("hour",                  IntegerType(), True),
        StructField("weekday",               IntegerType(), True),
        StructField("minute",                IntegerType(), True),
        StructField("is_rush_hour",          IntegerType(), True),
        StructField("vehicle_count",         IntegerType(), True),
        StructField("congestion_hour_score", DoubleType(),  True),
        StructField("is_peak_hour",          IntegerType(), True),
        StructField("estado",                StringType(),  True),
        StructField("fecha",                 StringType(),  True),
    ])
    return spark.createDataFrame(pd.DataFrame(rows), schema=schema)


# ═══════════════════════════════════════════════════════════════
#  Exportación → sklearn .joblib
# ═══════════════════════════════════════════════════════════════

def export_sklearn(spark_pipeline, features_df):
    print("\n📦 Exportando a sklearn (.joblib)...")

    labeled_df = spark_pipeline.stages[0].transform(features_df)
    rows = labeled_df.select(FEATURE_COLS + ["label", "sample_weight"]).collect()
    X       = np.array([[getattr(r, c) for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    y_spark = np.array([int(r["label"]) for r in rows])
    weights = np.array([float(r["sample_weight"]) for r in rows])

    label_stage  = spark_pipeline.stages[0]
    labels_order = label_stage.labels

    le = LabelEncoder()
    le.classes_ = np.array(labels_order)
    y_sk = y_spark.copy()

    rf_stage = spark_pipeline.stages[-1]
    n_trees  = rf_stage.getNumTrees
    depth    = rf_stage.getOrDefault("maxDepth")
    print(f"   numTrees={n_trees}  maxDepth={depth}  muestras={len(X)}")

    rf_sk = SklearnRF(
        n_estimators = n_trees,
        max_depth    = depth,
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1,
    )
    rf_sk.fit(X, y_sk, sample_weight=weights)

    joblib.dump(rf_sk, MODELS_DIR / "traffic_model.joblib")
    joblib.dump(le,    MODELS_DIR / "label_encoder.joblib")
    (MODELS_DIR / "feature_names.json").write_text(json.dumps(FEATURE_COLS, indent=2))

    print(f"   ✅ traffic_model.joblib")
    print(f"   ✅ label_encoder.joblib  → clases: {list(le.classes_)}")
    print(f"   ✅ feature_names.json    → {len(FEATURE_COLS)} features")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 58)
    print("  Entrenamiento MLlib — Rotonda Anchieta TF-24")
    print("  3 ROIs · 22 features · RandomForest + CrossValidator")
    print("=" * 58)

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("TrafficClassifier-TF24-Train")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"\n✅ SparkSession lista — cores: {spark.sparkContext.defaultParallelism}")

    # 1. Catálogo
    print("\n📂 Leyendo dataset...")
    catalog = build_catalog(spark)
    catalog.cache()
    print(f"   Distribución de clases:")
    catalog.groupBy("estado").count().orderBy("estado").show()

    # 2. Extracción distribuida de features
    print("🔄 Extrayendo features (mapPartitions + OpenCV)...")
    features_rdd = (
        catalog
        .repartition(spark.sparkContext.defaultParallelism)
        .rdd
        .mapPartitions(process_partition)
    )
    features_df = spark.createDataFrame(features_rdd, schema=_feature_schema())
    features_df.cache()
    n = features_df.count()
    print(f"   ✅ {n} muestras × {len(FEATURE_COLS)} features")

    # 3. Añadir pesos de clase para balanceo
    features_df = add_sample_weights(features_df, label_col="estado")
    features_df.cache()

    # 4. Split temporal: último día → test
    fechas = sorted(r["fecha"] for r in features_df.select("fecha").distinct().collect())
    print(f"\n📅 Días: {fechas}")

    if len(fechas) > 1:
        test_fecha = fechas[-1]
        train_df   = features_df.filter(F.col("fecha") != test_fecha).cache()
        test_df    = features_df.filter(F.col("fecha") == test_fecha).cache()
        print(f"   Split temporal → train: {train_df.count()}  test: {test_df.count()} ({test_fecha})")
    else:
        train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=42)
        train_df.cache(); test_df.cache()
        print("   ⚠️  Solo 1 día — split aleatorio 80/20")

    # 5. Pipeline MLlib
    print("\n🔧 Pipeline MLlib...")
    label_indexer = StringIndexer(
        inputCol="estado", outputCol="label",
        stringOrderType="frequencyDesc",
    )
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS, outputCol="features",
        handleInvalid="skip",
    )
    rf = SparkRF(
        labelCol="label", featuresCol="features",
        weightCol="sample_weight",
        numTrees=200, maxDepth=12,
        featureSubsetStrategy="sqrt", seed=42,
    )
    pipeline = Pipeline(stages=[label_indexer, assembler, rf])

    # 6. CrossValidator 5-fold
    print("🔍 CrossValidator 5-fold...")
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees, [100, 200])
        .addGrid(rf.maxDepth,  [8, 12])
        .build()
    )
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy",
    )
    cv = CrossValidator(
        estimator=pipeline, estimatorParamMaps=param_grid,
        evaluator=evaluator, numFolds=5, seed=42,
    )
    cv_model   = cv.fit(train_df)
    best_model = cv_model.bestModel
    best_rf    = best_model.stages[-1]

    print(f"\n   Mejor combinación:")
    print(f"     numTrees = {best_rf.getNumTrees}")
    print(f"     maxDepth = {best_rf.getOrDefault('maxDepth')}")
    print(f"   CV accuracy por fold:")
    for i, s in enumerate(cv_model.avgMetrics):
        print(f"     fold {i+1}: {s:.4f}")

    # 7. Evaluación en test
    print("\n📊 Evaluación en test...")
    preds = best_model.transform(test_df)
    acc   = evaluator.evaluate(preds)
    print(f"   Accuracy : {acc:.4f} ({acc*100:.1f}%)")
    for metric in ["weightedPrecision", "weightedRecall", "f1"]:
        evaluator.setMetricName(metric)
        print(f"   {metric:20s}: {evaluator.evaluate(preds):.4f}")

    print("\n   Matriz de confusión:")
    preds.groupBy("label", "prediction").count().orderBy("label", "prediction").show()

    print("   Top 10 features más importantes:")
    imp = sorted(
        zip(FEATURE_COLS, best_rf.featureImportances.toArray()),
        key=lambda x: x[1], reverse=True,
    )
    for feat, val in imp[:10]:
        print(f"     {feat:40s} {val:.4f}  {'█' * int(val * 200)}")

    # 8. Reentrenar con todos los datos y exportar
    print("\n🔁 Reentrenando con todos los datos...")
    final_pipeline = Pipeline(stages=[
        StringIndexer(inputCol="estado", outputCol="label", stringOrderType="frequencyDesc"),
        VectorAssembler(inputCols=FEATURE_COLS, outputCol="features", handleInvalid="skip"),
        SparkRF(
            labelCol="label", featuresCol="features",
            weightCol="sample_weight",
            numTrees=best_rf.getNumTrees,
            maxDepth=best_rf.getOrDefault("maxDepth"),
            featureSubsetStrategy="sqrt", seed=42,
        ),
    ])
    final_model = final_pipeline.fit(features_df)
    export_sklearn(final_model, features_df)

    print("\n" + "=" * 58)
    print("  ✅ Entrenamiento completado")
    print(f"     Modelos en: {MODELS_DIR}")
    print("=" * 58)
    print("\n  Aplica el modelo:")
    print("  docker compose restart backend")

    spark.stop()


if __name__ == "__main__":
    main()
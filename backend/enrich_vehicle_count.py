"""
enrich_vehicle_count.py — Añade vehicle_count a los labels.csv del dataset
===========================================================================
Lee los conteos pre-calculados por yolo_preview.py (yolo/counts.json)
y los escribe en labels.csv. Si no existe counts.json para un día,
ejecuta YOLO directamente como fallback.

Ejecutar dentro del contenedor backend:
  docker exec rotonda_backend python /app/enrich_vehicle_count.py

Requisitos:
  - /data/dataset montado (bind mount del compose)
  - /app/models/yolov8s.pt disponible (solo para el fallback)
  - labels.csv existente en cada carpeta de día
"""

import os
import cv2
import csv
import json
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

# ─── Rutas ────────────────────────────────────────────────────
DATASET_DIR = Path("/data/dataset")
MODEL_PATH  = os.environ.get("YOLO_MODEL_PATH", "/app/models/yolov8s.pt")

# ─── Geometría ────────────────────────────────────────────────
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
    'tramo1_rotonda': (413, 246, 640, 404),
    'tramo2_rotonda': (184, 200, 409, 337),
    'tramo3_rotonda': (3,   158, 179, 296),
}

VEHICLE_CLASSES   = {2, 3, 5, 7}
OVERLAP_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45

CSV_FIELDS = ['filename', 'timestamp', 'weekday', 'weekday_name',
              'hour', 'minute', 'is_rush_hour', 'estado', 'vehicle_count']


# ─── Helpers YOLO (fallback) ──────────────────────────────────

def _letterbox(img):
    h, w   = img.shape[:2]
    scale  = min(TARGET_W / w, TARGET_H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((TARGET_H, TARGET_W, 3), 114, dtype=np.uint8)
    px, py  = (TARGET_W - nw) // 2, (TARGET_H - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, px, py, scale, w, h


def _build_mask(px, py, scale, orig_w, orig_h):
    pts = MASK_POLYGON.astype(np.float32).copy()
    pts[:, 0] = pts[:, 0] * (orig_w / TARGET_W) * scale + px
    pts[:, 1] = pts[:, 1] * (orig_h / TARGET_H) * scale + py
    pts = np.clip(pts, 0, [TARGET_W - 1, TARGET_H - 1]).astype(np.int32)
    mask = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _adapt_rois(px, py, scale, orig_w, orig_h):
    adapted = {}
    for name, (x1, y1, x2, y2) in ROIS.items():
        adapted[name] = (
            max(0,            int(x1 * (orig_w / TARGET_W) * scale + px)),
            max(0,            int(y1 * (orig_h / TARGET_H) * scale + py)),
            min(TARGET_W - 1, int(x2 * (orig_w / TARGET_W) * scale + px)),
            min(TARGET_H - 1, int(y2 * (orig_h / TARGET_H) * scale + py)),
        )
    return adapted


def _overlap_ratio(box, roi):
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi
    ix1, iy1 = max(bx1, rx1), max(by1, ry1)
    ix2, iy2 = min(bx2, rx2), min(by2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / max((bx2 - bx1) * (by2 - by1), 1)


def _nms(boxes):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
    kept  = []
    for c in boxes:
        cx1, cy1, cx2, cy2, ccls, _ = c
        discard = False
        for k in kept:
            kx1, ky1, kx2, ky2, kcls, _ = k
            if ccls != kcls:
                continue
            ix1, iy1 = max(cx1, kx1), max(cy1, ky1)
            ix2, iy2 = min(cx2, kx2), min(cy2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            iou   = inter / ((cx2-cx1)*(cy2-cy1) + (kx2-kx1)*(ky2-ky1) - inter + 1e-6)
            if iou > NMS_IOU_THRESHOLD:
                discard = True
                break
        if not discard:
            kept.append(c)
    return kept


def count_vehicles_yolo(img_path: Path, model: YOLO) -> int:
    """Fallback: ejecuta YOLO directamente si no hay counts.json."""
    img = cv2.imread(str(img_path))
    if img is None:
        return -1

    canvas, px, py, scale, orig_w, orig_h = _letterbox(img)
    road_mask    = _build_mask(px, py, scale, orig_w, orig_h)
    adapted_rois = _adapt_rois(px, py, scale, orig_w, orig_h)

    masked = canvas.copy()
    masked[road_mask == 0] = 0

    with torch.no_grad():
        results = model(masked, verbose=False, conf=0.20, iou=0.45, device='cpu')[0]

    boxes = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2, cls, float(box.conf[0])))
    boxes = _nms(boxes)

    assigned = set()
    for roi_coords in adapted_rois.values():
        for i, (x1, y1, x2, y2, cls, conf) in enumerate(boxes):
            if i in assigned:
                continue
            if _overlap_ratio((x1, y1, x2, y2), roi_coords) >= OVERLAP_THRESHOLD:
                assigned.add(i)

    return len(assigned)


# ─── Procesamiento por día ────────────────────────────────────

def enrich_day(day_dir: Path, model=None) -> dict:
    csv_path   = day_dir / "labels.csv"
    frames_dir = day_dir / "frames"
    counts_path = day_dir / "yolo" / "counts.json"

    if not csv_path.exists():
        print(f"  ⚠️  Sin labels.csv en {day_dir.name}, saltando")
        return {}

    # Intentar leer conteos pre-calculados por yolo_preview.py
    if counts_path.exists():
        precomputed = json.loads(counts_path.read_text())
        print(f"  📅 {day_dir.name}: usando counts.json ({len(precomputed)} conteos)")
        use_precomputed = True
    else:
        precomputed = {}
        use_precomputed = False
        print(f"  📅 {day_dir.name}: sin counts.json — ejecutando YOLO (fallback)")
        if model is None:
            print(f"    ⚠️  Sin modelo YOLO disponible, saltando día")
            return {}

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
        original_fields = reader.fieldnames or []

    if 'vehicle_count' in original_fields:
        print(f"     ℹ️  Ya tiene vehicle_count — sobreescribiendo")

    total  = len(rows)
    errors = 0
    counts = {}

    for i, row in enumerate(rows, 1):
        fname    = row['filename']
        img_path = frames_dir / fname

        if use_precomputed:
            count = precomputed.get(fname, 0)
        else:
            count = count_vehicles_yolo(img_path, model)
            if count == -1:
                print(f"    ⚠️  [{i}/{total}] {fname} — no se pudo leer")
                count  = 0
                errors += 1

        row['vehicle_count'] = count
        counts[fname]        = count

        if i % 50 == 0 or i == total:
            print(f"    [{i}/{total}]  último conteo: {count}")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = day_dir / "labels.json"
    if json_path.exists():
        json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    avg = sum(counts.values()) / max(len(counts), 1)
    src = "precomputado" if use_precomputed else "YOLO directo"
    print(f"    ✅ {total - errors}/{total} procesadas  |  media: {avg:.1f} vehículos  |  fuente: {src}")

    return counts


# ─── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  Enriquecimiento de labels con vehicle_count")
    print(f"  Dataset: {DATASET_DIR}")
    print("=" * 58)

    if not DATASET_DIR.exists():
        print(f"\n❌ No se encontró {DATASET_DIR}")
        return

    days = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())
    if not days:
        print("❌ No hay carpetas de días en el dataset")
        return

    # Cargar YOLO solo si algún día no tiene counts.json
    needs_yolo = any(not (d / 'yolo' / 'counts.json').exists() for d in days)
    model = None
    if needs_yolo:
        print(f"\n📦 Cargando modelo YOLO (fallback para días sin counts.json)...")
        model = YOLO(MODEL_PATH)
        model.to('cpu')
        print("   ✅ Modelo listo\n")
    else:
        print("\n✅ Todos los días tienen counts.json — no se necesita YOLO\n")

    print(f"📂 Días encontrados: {[d.name for d in days]}\n")

    total_counts = {}
    for day_dir in days:
        counts = enrich_day(day_dir, model)
        total_counts.update(counts)
        print()

    print("=" * 58)
    print(f"  ✅ Completado — {len(total_counts)} imágenes enriquecidas")
    print("=" * 58)
    print("\n  Siguiente paso: reentrenar el modelo Spark")
    print("  docker exec rotonda_spark \\")
    print("    /opt/spark/bin/spark-submit \\")
    print("      --master local[*] \\")
    print("      --driver-memory 6g \\")
    print("      --conf spark.executor.memory=6g \\")
    print("      --conf spark.memory.fraction=0.8 \\")
    print("      --conf spark.sql.shuffle.partitions=4 \\")
    print("      /opt/spark_jobs/train_spark.py \\")
    print("    2>/dev/null | tee /tmp/train_output.txt")


if __name__ == "__main__":
    main()
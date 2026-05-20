#!/usr/bin/env python3
"""
yolo_preview.py — Genera imágenes anotadas con detecciones YOLO
================================================================
Lee los frames de dataset/202605XX/frames/, ejecuta YOLO con la
misma geometría que el backend, dibuja los bboxes de vehículos
dentro de la máscara y guarda el resultado en dataset/202605XX/yolo/.

También genera dataset/202605XX/yolo/counts.json con el conteo
por imagen, para que enrich_vehicle_count.py no tenga que
re-ejecutar YOLO.

Uso:
  python tools/yolo_preview.py --fecha 20260507
  python tools/yolo_preview.py --fecha 20260507 --modelo /ruta/yolov8s.pt

Requisitos (host, fuera de Docker):
  pip install ultralytics opencv-python
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ─── Rutas ───────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / 'backend').exists() else SCRIPT_DIR.parent
DATASET_DIR  = PROJECT_ROOT / 'dataset'
DEFAULT_MODEL = PROJECT_ROOT / 'backend' / 'models' / 'yolov8s.pt'

# ─── Geometría (idéntica a backend/detector.py) ───────────────
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

VEHICLE_CLASSES   = {2, 3, 5, 7}   # car, motorcycle, bus, truck
OVERLAP_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45

# Colores para dibujo (BGR)
COLOR_MASK    = (0,   60,   0)    # verde muy oscuro para área de máscara
COLOR_ROI     = {
    'tramo1_rotonda': (255, 100,   0),
    'tramo2_rotonda': (0,   200, 255),
    'tramo3_rotonda': (180,   0, 255),
}
COLOR_BOX     = (0, 255,  80)     # verde para vehículos detectados
COLOR_BOX_OUT = (80, 80,  80)     # gris para vehículos fuera de ROI
COLOR_TEXT_BG = (0,   0,   0)
COLOR_COUNT   = (0, 255, 180)


# ─── Helpers de geometría ─────────────────────────────────────

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
    return mask, pts


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


# ─── Procesamiento de una imagen ─────────────────────────────

def process_image(img_path: Path, model: YOLO) -> tuple[np.ndarray, int]:
    """
    Devuelve (imagen_anotada, vehicle_count).
    La imagen anotada tiene:
      - Contorno de la máscara en verde
      - Rectángulos de cada ROI con su nombre
      - Bboxes de vehículos (verde si están dentro de un ROI, gris si no)
      - Contador total en la esquina superior izquierda
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, -1

    canvas, px, py, scale, orig_w, orig_h = _letterbox(img)
    road_mask, mask_pts = _build_mask(px, py, scale, orig_w, orig_h)
    adapted_rois        = _adapt_rois(px, py, scale, orig_w, orig_h)

    # Ejecutar YOLO solo sobre zona enmascarada
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

    # Determinar qué vehículos están dentro de algún ROI
    assigned = set()
    for roi_coords in adapted_rois.values():
        for i, (x1, y1, x2, y2, cls, conf) in enumerate(boxes):
            if i in assigned:
                continue
            if _overlap_ratio((x1, y1, x2, y2), roi_coords) >= OVERLAP_THRESHOLD:
                assigned.add(i)

    vehicle_count = len(assigned)

    # ── Dibujar sobre el canvas original (sin enmascarar) ──

    # 1. Área de máscara semitransparente
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [mask_pts], (0, 40, 0))
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    # 2. Contorno de la máscara
    cv2.polylines(canvas, [mask_pts], isClosed=True, color=(0, 200, 0), thickness=2)

    # 3. Rectángulos de ROI con etiqueta
    for roi_name, (rx1, ry1, rx2, ry2) in adapted_rois.items():
        color = COLOR_ROI[roi_name]
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), color, 1)
        label = roi_name.replace('_rotonda', '')
        cv2.putText(canvas, label, (rx1 + 4, ry1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    # 4. Bboxes de vehículos
    cls_names = {2: 'car', 3: 'moto', 5: 'bus', 7: 'truck'}
    for i, (x1, y1, x2, y2, cls, conf) in enumerate(boxes):
        in_roi = i in assigned
        color  = COLOR_BOX if in_roi else COLOR_BOX_OUT
        thick  = 2 if in_roi else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
        txt = f"{cls_names.get(cls, str(cls))} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(canvas, txt, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    # 5. Contador total en esquina superior izquierda
    count_txt = f"Vehiculos: {vehicle_count}"
    (cw, ch), _ = cv2.getTextSize(count_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(canvas, (6, 6), (cw + 16, ch + 16), (0, 0, 0), -1)
    cv2.putText(canvas, count_txt, (10, ch + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_COUNT, 2, cv2.LINE_AA)

    return canvas, vehicle_count


# ─── Main ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Genera imágenes anotadas con YOLO")
    p.add_argument('--fecha',   required=True, help='YYYYMMDD')
    p.add_argument('--modelo',  type=str, default=str(DEFAULT_MODEL),
                   help=f'Ruta al modelo YOLO (default: {DEFAULT_MODEL})')
    p.add_argument('--proyecto', type=str, default=None,
                   help='Ruta raíz del proyecto (si se ejecuta desde otro directorio)')
    return p.parse_args()


def main():
    args = parse_args()

    global DATASET_DIR
    if args.proyecto:
        DATASET_DIR = Path(args.proyecto).resolve() / 'dataset'

    day_dir    = DATASET_DIR / args.fecha
    frames_dir = day_dir / 'frames'
    yolo_dir   = day_dir / 'yolo'
    yolo_dir.mkdir(parents=True, exist_ok=True)

    if not frames_dir.exists():
        print(f"❌ No se encontró {frames_dir}")
        print("   Ejecuta primero el etiquetador para descargar los frames.")
        sys.exit(1)

    frame_list = sorted(frames_dir.glob('*.jpg'))
    if not frame_list:
        print(f"❌ No hay imágenes en {frames_dir}")
        sys.exit(1)

    model_path = Path(args.modelo)
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        print("   Indica la ruta con --modelo /ruta/yolov8s.pt")
        sys.exit(1)

    print("=" * 58)
    print("  YOLO Preview — Rotonda Anchieta TF-24")
    print(f"  Día     : {args.fecha}")
    print(f"  Frames  : {len(frame_list)}")
    print(f"  Modelo  : {model_path.name}")
    print(f"  Salida  : {yolo_dir}")
    print("=" * 58)

    print("\n📦 Cargando modelo YOLO...")
    model = YOLO(str(model_path))
    print("   ✅ Listo\n")

    counts = {}
    errors = 0

    for i, img_path in enumerate(frame_list, 1):
        annotated, count = process_image(img_path, model)

        if annotated is None:
            print(f"  ⚠️  [{i}/{len(frame_list)}] {img_path.name} — error de lectura")
            counts[img_path.name] = 0
            errors += 1
            continue

        out_path = yolo_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)
        counts[img_path.name] = count

        if i % 50 == 0 or i == len(frame_list):
            print(f"  [{i}/{len(frame_list)}]  {img_path.name}  → {count} vehículos")

    # Guardar counts.json para que enrich_vehicle_count.py lo reutilice
    counts_path = yolo_dir / 'counts.json'
    counts_path.write_text(json.dumps(counts, indent=2, ensure_ascii=False))

    total_ok = len(frame_list) - errors
    avg      = sum(counts.values()) / max(total_ok, 1)
    print(f"\n✅ {total_ok}/{len(frame_list)} imágenes procesadas")
    print(f"   Media de vehículos por frame: {avg:.1f}")
    print(f"   Conteos guardados en: yolo/counts.json")
    print(f"\n  Siguiente paso:")
    print(f"  python tools/etiquetador_manual.py --fecha {args.fecha} --sin-drive")


if __name__ == '__main__':
    main()
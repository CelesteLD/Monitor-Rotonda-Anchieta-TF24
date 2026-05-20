"""
test_detector.py — Diagnóstico de conteo de vehículos
======================================================
Ejecutar:
    python3 test_detector.py --images ./images/ --model ./models/yolov8n.pt

Por cada imagen genera en images/outputs/:
  - <nombre>_prod.jpg   → Vista "producción": lo que vería el usuario en el frontend
  - <nombre>_debug.jpg  → Panel 4 cuadrantes de diagnóstico

Uso con modelo alternativo:
    python3 test_detector.py --images ./images/ --model ./models/yolov8s.pt --conf 0.20
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from collections import Counter

# ──────────────────────────────────────────────────────────────
# Configuración (igual que detector.py)
# ──────────────────────────────────────────────────────────────

ROAD_MASK_POLYGON = np.array([(638, 304), (539, 274), (495, 262), (459, 249), 
                              (437, 245), (411, 244), (386, 239), (365, 228), 
                              (349, 226), (323, 222), (299, 221), (277, 221), 
                              (245, 222), (227, 222), (209, 216), (194, 210), 
                              (168, 204), (150, 204), (136, 203), (121, 197), 
                              (106, 196), (83, 192), (68, 199), (37, 203), (17, 206), 
                              (7, 209), (5, 275), (94, 274), (141, 289), (209, 305), 
                              (272, 313), (331, 327), (368, 346), (418, 361), (473, 381), 
                              (508, 389), (536, 396), (577, 410), (601, 411), 
                              (624, 419), (633, 420), (639, 423)], 
                              dtype=np.int32)

ROIS = {
    'tramo1_rotonda': (413, 246, 640, 404),
    'tramo2_rotonda': (184, 200, 409, 337),
    'tramo3_rotonda': (3, 158, 179, 296),
}

ROI_COLORS = {
    'tramo1_rotonda':   (  0, 200, 255),
    'tramo2_rotonda':   (  0, 140, 255),
    'tramo3_rotonda':   (255, 140,   0),
}

ROI_LABELS = {
    'tramo1_rotonda':   'Tramo 1',
    'tramo2_rotonda':   'Tramo 2',
    'tramo3_rotonda':   'Tramo 3',
}

VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_COLORS  = {
    2: (255, 200,   0),
    3: (255, 100, 200),
    5: (  0, 200, 255),
    7: (  0,  80, 255),
}

OVERLAP_THRESHOLD = 0.25

# ──────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────

def build_road_mask():
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.fillPoly(mask, [ROAD_MASK_POLYGON], 255)
    return mask

def overlap_ratio(box, roi):
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi
    ix1, iy1 = max(bx1, rx1), max(by1, ry1)
    ix2, iy2 = min(bx2, rx2), min(by2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / max((bx2 - bx1) * (by2 - by1), 1)

def assign_roi_exclusive(vehicles):
    """
    Asigna cada vehículo a UNA sola ROI (la de mayor overlap >= threshold).
    Evita el doble conteo cuando un vehículo está en el límite de dos ROIs.
    Devuelve dict roi_name -> lista de vehículos asignados.
    """
    roi_vehicle_map = {k: [] for k in ROIS}
    for v in vehicles:
        x1, y1, x2, y2 = v[0], v[1], v[2], v[3]
        best_roi_name, best_r = None, 0.0
        for roi_name, roi_coords in ROIS.items():
            r = overlap_ratio((x1, y1, x2, y2), roi_coords)
            if r > best_r:
                best_roi_name, best_r = roi_name, r
        if best_roi_name and best_r >= OVERLAP_THRESHOLD:
            roi_vehicle_map[best_roi_name].append(v)
    return roi_vehicle_map

# ──────────────────────────────────────────────────────────────
# Imagen de producción
# ──────────────────────────────────────────────────────────────

def draw_production_frame(frame_resized, roi_vehicle_map):
    """Genera la imagen tal como aparecería en el frontend."""
    annotated = frame_resized.copy()

    cv2.polylines(annotated, [ROAD_MASK_POLYGON], True, (70, 70, 70), 1)

    total = sum(len(v) for v in roi_vehicle_map.values())

    overlay = annotated.copy()
    for name, (x1, y1, x2, y2) in ROIS.items():
        count = len(roi_vehicle_map[name])
        color = ROI_COLORS[name]
        alpha = 0.10 + min(count * 0.06, 0.30)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        overlay = annotated.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f'{ROI_LABELS[name]}: {count}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(annotated,
                      (x1 + 3, y1 + 2), (x1 + 3 + tw + 4, y1 + th + 8),
                      (0, 0, 0), -1)
        cv2.putText(annotated, label, (x1 + 5, y1 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    for name, vlist in roi_vehicle_map.items():
        for (x1, y1, x2, y2, cls, conf) in vlist:
            color = VEHICLE_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f'{VEHICLE_CLASSES[cls]} {conf:.2f}'
            cv2.putText(annotated, label, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    estado_color = (0, 200, 0) if total < 5 else \
                   (0, 165, 255) if total < 10 else (0, 0, 255)
    cv2.rectangle(annotated, (6, 6), (230, 44), (0, 0, 0), -1)
    cv2.putText(annotated, f'Vehiculos: {total}',
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, estado_color, 2, cv2.LINE_AA)

    return annotated, total

# ──────────────────────────────────────────────────────────────
# Panel de debug (4 cuadrantes)
# ──────────────────────────────────────────────────────────────

def draw_debug_panel(frame_resized, masked, vehicles, non_vehicles,
                     roi_vehicle_map, coco_names, conf_threshold):
    h, w = frame_resized.shape[:2]
    canvas = np.zeros((h * 2 + 10, w * 2 + 10, 3), dtype=np.uint8)

    # TL: frame original + ROIs
    tl = frame_resized.copy()
    cv2.polylines(tl, [ROAD_MASK_POLYGON], True, (80, 80, 80), 1)
    for name, (x1, y1, x2, y2) in ROIS.items():
        cv2.rectangle(tl, (x1, y1), (x2, y2), ROI_COLORS[name], 2)
        cv2.putText(tl, ROI_LABELS[name], (x1 + 3, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ROI_COLORS[name], 1)
    cv2.putText(tl, 'ORIGINAL + ROIs', (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    canvas[:h, :w] = tl

    # TR: imagen enmascarada
    tr = masked.copy()
    cv2.putText(tr, f'LO QUE VE YOLO (conf>={conf_threshold})', (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    canvas[:h, w + 10:w + 10 + w] = tr

    # BL: detecciones aceptadas con ROI exclusiva
    bl = frame_resized.copy()
    cv2.polylines(bl, [ROAD_MASK_POLYGON], True, (60, 60, 60), 1)
    for name, (rx1, ry1, rx2, ry2) in ROIS.items():
        count = len(roi_vehicle_map[name])
        cv2.rectangle(bl, (rx1, ry1), (rx2, ry2), ROI_COLORS[name], 2)
        cv2.putText(bl, f'{ROI_LABELS[name]}: {count}', (rx1 + 3, ry1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ROI_COLORS[name], 1)
    for name, vlist in roi_vehicle_map.items():
        for (x1, y1, x2, y2, cls, conf) in vlist:
            color = VEHICLE_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(bl, (x1, y1), (x2, y2), color, 2)
            cv2.putText(bl, f'{VEHICLE_CLASSES[cls]} {conf:.2f}',
                        (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    total = sum(len(v) for v in roi_vehicle_map.values())
    estado = (0, 200, 0) if total < 5 else (0, 165, 255) if total < 10 else (0, 0, 255)
    cv2.putText(bl, f'Total (excl.): {total}', (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, estado, 2)
    canvas[h + 10:, :w] = bl

    # BR: descartados
    br = frame_resized.copy()
    cv2.putText(br, 'DESCARTADOS (no-clase / sin ROI)', (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 255), 1)
    for (x1, y1, x2, y2, cls, conf) in non_vehicles:
        cv2.rectangle(br, (x1, y1), (x2, y2), (80, 80, 255), 1)
        cv2.putText(br, f'{coco_names.get(cls, "?")} {conf:.2f}',
                    (x1, max(y1 - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150, 150, 255), 1)
    assigned_ids = {id(v) for vlist in roi_vehicle_map.values() for v in vlist}
    for v in vehicles:
        if id(v) not in assigned_ids:
            x1, y1, x2, y2, cls, conf = v
            cv2.rectangle(br, (x1, y1), (x2, y2), (0, 80, 200), 2)
            cv2.putText(br, f'SIN ROI {conf:.2f}',
                        (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 100, 220), 1)
    canvas[h + 10:, w + 10:w + 10 + w] = br

    return canvas

# ──────────────────────────────────────────────────────────────
# Diagnóstico por imagen
# ──────────────────────────────────────────────────────────────

def diagnose_image(img_path: Path, model: YOLO, road_mask: np.ndarray,
                   conf_threshold: float, out_dir: Path):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  [!] No se pudo leer: {img_path}")
        return

    frame_resized = cv2.resize(frame, (640, 480))

    masked = frame_resized.copy()
    masked[road_mask == 0] = 0

    with torch.no_grad():
        results = model(masked, verbose=False, conf=conf_threshold, device='cpu')[0]

    all_detections = []
    for box in results.boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        all_detections.append((x1, y1, x2, y2, cls, conf))

    vehicles     = [d for d in all_detections if d[4] in VEHICLE_CLASSES]
    non_vehicles = [d for d in all_detections if d[4] not in VEHICLE_CLASSES]

    roi_vehicle_map = assign_roi_exclusive(vehicles)
    roi_counts      = {k: len(v) for k, v in roi_vehicle_map.items()}
    total           = sum(roi_counts.values())
    coco_names      = results.names

    prod_frame, _ = draw_production_frame(frame_resized, roi_vehicle_map)
    prod_path = out_dir / f"{img_path.stem}_prod.jpg"
    cv2.imwrite(str(prod_path), prod_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    debug_canvas = draw_debug_panel(
        frame_resized, masked, vehicles, non_vehicles,
        roi_vehicle_map, coco_names, conf_threshold)
    debug_path = out_dir / f"{img_path.stem}_debug.jpg"
    cv2.imwrite(str(debug_path), debug_canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])

    assigned_ids = {id(v) for vlist in roi_vehicle_map.values() for v in vlist}
    outside_roi  = len([v for v in vehicles if id(v) not in assigned_ids])

    print(f"\n{'='*60}")
    print(f"  Imagen : {img_path.name}")
    print(f"  Conf   : {conf_threshold}")
    print(f"{'='*60}")
    print(f"  Total YOLO       : {len(all_detections)}")
    print(f"  Vehículos        : {len(vehicles)}")
    print(f"  No-vehículo      : {len(non_vehicles)}")
    print(f"  Sin ROI asignada : {outside_roi}")
    print(f"\n  Conteo por ROI:")
    for name, count in roi_counts.items():
        bar = '█' * count
        print(f"    {ROI_LABELS[name]:<18}: {count:>2}  {bar}")
    print(f"  {'TOTAL':<18}: {total:>2}")

    if non_vehicles:
        print(f"\n  Clases ignoradas:")
        for cls_name, cnt in Counter(
                coco_names.get(d[4], '?') for d in non_vehicles).most_common():
            print(f"    {cls_name}: {cnt}")

    print(f"\n  → {prod_path.name}")
    print(f"  → {debug_path.name}")

    return {
        'image': img_path.name,
        'total': total,
        'roi_counts': roi_counts,
        'yolo_total': len(all_detections),
        'vehicles': len(vehicles),
    }

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--model',  default='yolov8n.pt')
    parser.add_argument('--conf',   type=float, default=0.20)
    parser.add_argument('--out',    default=None,
                        help='Carpeta de salida (por defecto: <images>/outputs/)')
    args = parser.parse_args()

    img_dir = Path(args.images)
    out_dir = Path(args.out) if args.out else img_dir / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    exts   = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    if not images:
        print(f"[!] No se encontraron imágenes en {img_dir}")
        sys.exit(1)

    print(f"[+] Modelo  : {args.model}")
    model = YOLO(args.model)
    model.to('cpu')
    road_mask = build_road_mask()

    print(f"[+] Imágenes: {len(images[:10])}  |  conf={args.conf}")
    print(f"[+] Salida  : {out_dir}/")

    results = []
    for img_path in images[:10]:
        r = diagnose_image(img_path, model, road_mask, args.conf, out_dir)
        if r:
            results.append(r)

    print(f"\n{'='*60}")
    print("  RESUMEN")
    print(f"{'='*60}")
    print(f"  {'Imagen':<30} {'YOLO':>6} {'Veh':>5} {'Contado':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*5} {'-'*8}")
    for r in results:
        print(f"  {r['image']:<30} {r['yolo_total']:>6} {r['vehicles']:>5} {r['total']:>8}")

    avg = sum(r['total'] for r in results) / max(len(results), 1)
    print(f"\n  Media vehículos : {avg:.1f}")
    print(f"  Salida          : {out_dir}/")

if __name__ == '__main__':
    main()
import os
import cv2
import numpy as np
from joblib import Parallel, delayed
from services.roi_counter import count_boxes_per_roi   # ← nuevo

# ── Máscara de carretera (carriles interiores de la rotonda) ───
_MASK_POLYGON_640 = np.array([
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

# ROIs calibradas en espacio 640×480
ROIS = {
    'tramo1_rotonda': (413, 246, 640, 404),
    'tramo2_rotonda': (184, 200, 409, 337),
    'tramo3_rotonda': (3,   158, 179, 296),
}

# Reemplazar ROI_COLORS
ROI_COLORS = {
    'tramo1_rotonda': (111,  71, 239),   # pink    #ef476f
    'tramo2_rotonda': (102, 209, 255),   # yellow  #ffd166
    'tramo3_rotonda': (160, 214,   6),   # emerald #06d6a0
}

ROI_LABELS = {
    'tramo1_rotonda': 'Tramo 1',
    'tramo2_rotonda': 'Tramo 2',
    'tramo3_rotonda': 'Tramo 3',
}

VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
VEHICLE_COLORS  = {
    2: (255, 200,   0),
    3: (255, 100, 200),
    5: (  0, 200, 255),
    7: (  0,  80, 255),
}

OVERLAP_THRESHOLD = 0.25
TARGET_W, TARGET_H = 640, 480

NMS_IOU_THRESHOLD = 0.45


# ── Letterbox ─────────────────────────────────────────────────
def _resize_letterbox(frame: np.ndarray):
    h, w   = frame.shape[:2]
    scale  = min(TARGET_W / w, TARGET_H / h)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x  = (TARGET_W - new_w) // 2
    pad_y  = (TARGET_H - new_h) // 2
    canvas = np.full((TARGET_H, TARGET_W, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, pad_x, pad_y, scale


def _adapt_polygon(polygon, pad_x, pad_y, scale, orig_w, orig_h):
    pts = polygon.astype(np.float32)
    pts[:, 0] = pts[:, 0] * (orig_w / TARGET_W) * scale + pad_x
    pts[:, 1] = pts[:, 1] * (orig_h / TARGET_H) * scale + pad_y
    return np.clip(pts, 0, [TARGET_W - 1, TARGET_H - 1]).astype(np.int32)


def _adapt_rois(pad_x, pad_y, scale, orig_w, orig_h):
    adapted = {}
    for name, (x1, y1, x2, y2) in ROIS.items():
        nx1 = int(x1 * (orig_w / TARGET_W) * scale + pad_x)
        ny1 = int(y1 * (orig_h / TARGET_H) * scale + pad_y)
        nx2 = int(x2 * (orig_w / TARGET_W) * scale + pad_x)
        ny2 = int(y2 * (orig_h / TARGET_H) * scale + pad_y)
        adapted[name] = (
            max(0, nx1), max(0, ny1),
            min(TARGET_W - 1, nx2), min(TARGET_H - 1, ny2),
        )
    return adapted


# ── NMS manual ────────────────────────────────────────────────
def _apply_nms(vehicle_boxes):
    if not vehicle_boxes:
        return []
    boxes = sorted(vehicle_boxes, key=lambda b: b[5], reverse=True)
    kept  = []
    for candidate in boxes:
        cx1, cy1, cx2, cy2, ccls, _ = candidate
        discard = False
        for kx1, ky1, kx2, ky2, kcls, _ in kept:
            if ccls != kcls:
                continue
            ix1 = max(cx1, kx1); iy1 = max(cy1, ky1)
            ix2 = min(cx2, kx2); iy2 = min(cy2, ky2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area_c = max((cx2 - cx1) * (cy2 - cy1), 1)
            area_k = max((kx2 - kx1) * (ky2 - ky1), 1)
            iou = inter / (area_c + area_k - inter)
            if iou > NMS_IOU_THRESHOLD:
                discard = True
                break
        if not discard:
            kept.append(candidate)
    return kept


# ── YOLO ───────────────────────────────────────────────────────
USE_YOLO = False
try:
    import torch
    from ultralytics import YOLO
    MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', '/app/models/yolov8s.pt')
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    _model = YOLO(MODEL_PATH)
    _model.to('cpu')
    USE_YOLO = True
    print('[Detector] YOLOv8 small cargado en CPU')
except Exception as e:
    print(f'[Detector] YOLO no disponible ({e}) — usando MOG2')

_bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=30, varThreshold=25, detectShadows=True
)

N_JOBS = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 4))


# ── Utilidades ─────────────────────────────────────────────────
def _overlap_ratio(box, roi):
    bx1, by1, bx2, by2 = box
    rx1, ry1, rx2, ry2 = roi
    ix1 = max(bx1, rx1); iy1 = max(by1, ry1)
    ix2 = min(bx2, rx2); iy2 = min(by2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / max((bx2 - bx1) * (by2 - by1), 1)


def _assign_roi_exclusive(vehicle_boxes, adapted_rois):
    """Asigna cada box a la ROI con mayor solapamiento (exclusivo)."""
    roi_vehicle_map = {k: [] for k in adapted_rois}
    for box in vehicle_boxes:
        x1, y1, x2, y2 = box[:4]
        best_roi, best_r = None, 0.0
        for roi_name, roi_coords in adapted_rois.items():
            r = _overlap_ratio((x1, y1, x2, y2), roi_coords)
            if r > best_r:
                best_roi, best_r = roi_name, r
        if best_roi and best_r >= OVERLAP_THRESHOLD:
            roi_vehicle_map[best_roi].append(box)
    return roi_vehicle_map


# ── Detección YOLO ─────────────────────────────────────────────
def _detect_yolo(canvas, road_mask, adapted_polygon, adapted_rois):
    masked = canvas.copy()
    masked[road_mask == 0] = 0

    import torch
    with torch.no_grad():
        results = _model(
            masked, verbose=False,
            conf=0.20,
            iou=0.45,
            device='cpu',
        )[0]

    # Recoger solo vehículos
    vehicle_boxes = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            vehicle_boxes.append((x1, y1, x2, y2, cls, conf))

    vehicle_boxes = _apply_nms(vehicle_boxes)

    # ── Conteo por ROI via binario C++ con OpenMP ──────────────
    boxes_for_cpp = [
        {'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3]}
        for b in vehicle_boxes
    ]
    cpp_result = count_boxes_per_roi(boxes_for_cpp, adapted_rois)

    # Extraer metadatos de rendimiento y construir roi_counts limpio
    elapsed_ms = cpp_result.pop('_elapsed_ms', 0.0)
    threads    = cpp_result.pop('_threads', 1)
    roi_counts = cpp_result   # { 'tramo1_rotonda': N, ... }

    # roi_vehicle_map solo para el dibujado (no para el conteo)
    roi_vehicle_map = _assign_roi_exclusive(vehicle_boxes, adapted_rois)

    # ── Dibujar ────────────────────────────────────────────────
    annotated = canvas.copy()
    cv2.polylines(annotated, [adapted_polygon], True, (70, 70, 70), 1)

    overlay = annotated.copy()
    for name, (x1, y1, x2, y2) in adapted_rois.items():
        count = roi_counts.get(name, 0)
        color = ROI_COLORS[name]
        alpha = 0.10 + min(count * 0.06, 0.30)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        overlay = annotated.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f'{ROI_LABELS[name]}: {count}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(annotated, (x1+3, y1+2), (x1+3+tw+4, y1+th+8), (0, 0, 0), -1)
        cv2.putText(annotated, label, (x1+5, y1+th+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    for name, vlist in roi_vehicle_map.items():
        for (x1, y1, x2, y2, cls, conf) in vlist:
            color_box = VEHICLE_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_box, 2)
            cv2.putText(annotated, f'{VEHICLE_CLASSES[cls]} {conf:.2f}',
                        (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, color_box, 1, cv2.LINE_AA)

    return roi_counts, annotated, elapsed_ms, threads


# ── Detección MOG2 (fallback sin YOLO) ────────────────────────
def _detect_mog2(canvas, road_mask, adapted_rois):
    gray = cv2.GaussianBlur(
        cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
    masked_gray = cv2.bitwise_and(gray, gray, mask=road_mask)
    fg_mask = _bg_subtractor.apply(masked_gray)

    def count_roi(roi):
        x1, y1, x2, y2 = roi
        region = fg_mask[y1:y2, x1:x2]
        _, mask = cv2.threshold(region, 200, 255, cv2.THRESH_BINARY)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len([c for c in contours if cv2.contourArea(c) > 400])

    counts = Parallel(n_jobs=N_JOBS, backend='threading')(
        delayed(count_roi)(roi) for roi in adapted_rois.values()
    )
    return dict(zip(adapted_rois.keys(), counts)), canvas.copy(), 0.0, 1


# ── Punto de entrada ───────────────────────────────────────────
def detect(image_bytes: bytes) -> dict:
    arr   = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {
            'vehicle_count': 0, 'roi_counts': {},
            'annotated_jpg': image_bytes, 'cuda_used': False,
            'cpp_elapsed_ms': 0.0, 'cpp_threads': 0,
        }

    orig_h, orig_w = frame.shape[:2]
    canvas, pad_x, pad_y, scale = _resize_letterbox(frame)

    adapted_polygon = _adapt_polygon(_MASK_POLYGON_640, pad_x, pad_y, scale, orig_w, orig_h)
    adapted_rois    = _adapt_rois(pad_x, pad_y, scale, orig_w, orig_h)

    road_mask = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
    cv2.fillPoly(road_mask, [adapted_polygon], 255)

    if USE_YOLO:
        roi_counts, annotated, elapsed_ms, threads = _detect_yolo(
            canvas, road_mask, adapted_polygon, adapted_rois)
    else:
        roi_counts, annotated, elapsed_ms, threads = _detect_mog2(
            canvas, road_mask, adapted_rois)

    vehicle_count   = sum(roi_counts.values())
    visual_features = extract_visual_features(canvas, road_mask, adapted_rois)

    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return {
        'vehicle_count':   vehicle_count,
        'roi_counts':      roi_counts,
        'annotated_jpg':   buf.tobytes(),
        'cuda_used':       False,
        'visual_features': visual_features,
        'cpp_elapsed_ms':  round(elapsed_ms, 3),   #  latencia del binario
        'cpp_threads':     threads,                 # threads OpenMP usados
    }


# ── Extracción de features visuales ───────────────────────────
def extract_visual_features(canvas: np.ndarray, road_mask: np.ndarray,
                             adapted_rois: dict) -> dict:
    hsv   = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray  = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = cv2.Canny(cv2.GaussianBlur(gray.astype(np.uint8), (3, 3), 0), 50, 150)

    features = {}
    for name, (x1, y1, x2, y2) in adapted_rois.items():
        roi_mask = road_mask[y1:y2, x1:x2]
        px_count = roi_mask.sum() / 255 + 1
        v_roi = hsv[y1:y2, x1:x2, 2]
        s_roi = hsv[y1:y2, x1:x2, 1]
        g_roi = gray[y1:y2, x1:x2]
        e_roi = edges[y1:y2, x1:x2]
        m     = roi_mask / 255

        features[f"{name}_brightness_mean"] = float(np.sum(v_roi * m) / px_count)
        features[f"{name}_brightness_std"]  = float(v_roi[roi_mask > 0].std() if roi_mask.any() else 0)
        features[f"{name}_saturation_mean"] = float(np.sum(s_roi * m) / px_count)
        features[f"{name}_edge_density"]    = float(np.sum(e_roi * m) / px_count / 255)
        features[f"{name}_gray_contrast"]   = float(g_roi[roi_mask > 0].std() if roi_mask.any() else 0)

    return features
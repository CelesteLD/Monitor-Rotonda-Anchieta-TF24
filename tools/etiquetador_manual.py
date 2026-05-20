#!/usr/bin/env python3
"""
etiquetador_manual.py — Etiquetado manual de frames de tráfico.

Muestra cada imagen y tú asignas el estado con el teclado.
Si existe dataset/202605XX/yolo/ muestra las imágenes anotadas
por YOLO (con bboxes y conteo), pero guarda el filename original
de frames/ en el CSV.

Controles:
  F  →  FLUIDO
  D  →  DENSO
  C  →  COLAPSO
  Z  →  Deshacer última etiqueta (volver un frame atrás)
  S  →  Saltar frame (revisar después)
  Q  →  Guardar progreso y salir

Uso:
  python etiquetador_manual.py --fecha 20260507
  python etiquetador_manual.py --fecha 20260507 --desde 80     # reanudar
  python etiquetador_manual.py --fecha 20260507 --sin-drive    # frames ya locales
"""

import os
import sys
import json
import csv
import argparse
import warnings
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np

warnings.filterwarnings('ignore')

# ── Rutas ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / 'backend').exists() else SCRIPT_DIR.parent
DATASET_DIR  = PROJECT_ROOT / 'dataset'
DRIVE_FOLDER = 'dataset_rotonda_anchieta'

ESTADOS = ['FLUIDO', 'DENSO', 'COLAPSO']

KEY_MAP = {
    ord('f'): 'FLUIDO',  ord('F'): 'FLUIDO',
    ord('d'): 'DENSO',   ord('D'): 'DENSO',
    ord('c'): 'COLAPSO', ord('C'): 'COLAPSO',
    ord('s'): 'SKIP',    ord('S'): 'SKIP',
    ord('z'): 'UNDO',    ord('Z'): 'UNDO',
    ord('q'): 'QUIT',    ord('Q'): 'QUIT',
}

ESTADO_COLOR = {
    'FLUIDO':  (40,  167,  69),
    'DENSO':   (30,  144, 255),
    'COLAPSO': (60,   60, 220),
}

CSV_FIELDS = ['filename', 'timestamp', 'weekday', 'weekday_name',
              'hour', 'minute', 'is_rush_hour', 'estado']


# ── Google Drive ───────────────────────────────────────────────────────────────
def get_drive_service():
    try:
        from googleapiclient.discovery import build
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
    except ImportError:
        print("⚠  Instala: pip install google-api-python-client google-auth-oauthlib")
        sys.exit(1)

    token_path = Path.home() / 'token.json'
    if not token_path.exists():
        print(f"❌ No se encontró {token_path}. Ejecuta el flujo OAuth primero.")
        sys.exit(1)

    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(
        str(token_path), scopes=['https://www.googleapis.com/auth/drive.file'])
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())
    return build('drive', 'v3', credentials=creds)


def get_folder_id(service, name):
    q = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    r = service.files().list(q=q, fields='files(id)').execute()
    files = r.get('files', [])
    if not files:
        print(f"❌ Carpeta '{name}' no encontrada en Drive.")
        sys.exit(1)
    return files[0]['id']


def download_day(service, folder_id, fecha_str, frames_dir: Path, meta_dir: Path):
    from googleapiclient.http import MediaIoBaseDownload

    q = (f"'{folder_id}' in parents and trashed=false "
         f"and name contains '{fecha_str}'")
    files, page_token = [], None
    while True:
        r = service.files().list(
            q=q, fields='nextPageToken,files(id,name)',
            pageSize=1000, pageToken=page_token).execute()
        files.extend(r.get('files', []))
        page_token = r.get('nextPageToken')
        if not page_token:
            break

    jpgs  = sorted([f for f in files if f['name'].endswith('.jpg')], key=lambda x: x['name'])
    jsons = {f['name'].replace('.jpg', '.json'): f
             for f in files if f['name'].endswith('.json')}

    if not jpgs:
        print(f"❌ No hay imágenes para el día {fecha_str} en Drive.")
        sys.exit(1)

    print(f"[Drive] {len(jpgs)} frames encontrados. Descargando...")
    for i, f in enumerate(jpgs, 1):
        dest = frames_dir / f['name']
        if not dest.exists():
            buf = BytesIO()
            dl  = MediaIoBaseDownload(buf, service.files().get_media(fileId=f['id']))
            done = False
            while not done:
                _, done = dl.next_chunk()
            dest.write_bytes(buf.getvalue())
            print(f"  [{i}/{len(jpgs)}] {f['name']} ({len(buf.getvalue())//1024} KB)")
        else:
            print(f"  [{i}/{len(jpgs)}] {f['name']} ya existe")

        meta_name = f['name'].replace('.jpg', '.json')
        if meta_name in jsons and not (meta_dir / meta_name).exists():
            buf = BytesIO()
            dl  = MediaIoBaseDownload(buf, service.files().get_media(fileId=jsons[meta_name]['id']))
            done = False
            while not done:
                _, done = dl.next_chunk()
            (meta_dir / meta_name).write_bytes(buf.getvalue())

    return sorted(frames_dir.glob('*.jpg'))


# ── UI ─────────────────────────────────────────────────────────────────────────
def draw_frame(img_bgr, fname, idx, total, meta, usando_yolo: bool):
    H, W = img_bgr.shape[:2]

    target_h = 520
    scale    = target_h / H
    display  = cv2.resize(img_bgr, (int(W * scale), target_h), interpolation=cv2.INTER_LINEAR)
    dH, dW   = display.shape[:2]

    panel_h = 90
    canvas  = np.zeros((dH + panel_h, dW, 3), dtype=np.uint8)
    canvas[:dH] = display
    canvas[dH:] = (18, 18, 28)

    cv2.line(canvas, (0, dH), (dW, dH), (50, 50, 60), 1)

    # Indicador de modo YOLO
    modo_txt = "👁 YOLO preview" if usando_yolo else "📷 Frame original"
    modo_col = (0, 220, 120) if usando_yolo else (160, 160, 170)
    cv2.putText(canvas, f'{fname}   [{idx}/{total}]   {modo_txt}',
                (10, dH + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, modo_col, 1, cv2.LINE_AA)

    if meta:
        hora = meta.get('hour', '?')
        dia  = meta.get('weekday_name', '')
        rush = ' · hora punta' if meta.get('is_rush_hour') else ''
        cv2.putText(canvas, f'{dia}  {hora:02d}h{rush}',
                    (10, dH + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 115), 1, cv2.LINE_AA)

    cv2.putText(canvas,
                '[F] Fluido    [D] Denso    [C] Colapso    [Z] Deshacer    [S] Saltar    [Q] Salir',
                (10, dH + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (75, 75, 90), 1, cv2.LINE_AA)

    return canvas


def draw_confirm(canvas, estado):
    color   = ESTADO_COLOR[estado]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], canvas.shape[0]), color, -1)
    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

    text  = estado
    scale = 1.8
    thick = 3
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cx = (canvas.shape[1] - tw) // 2
    cy = (canvas.shape[0] + th) // 2
    cv2.putText(canvas, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return canvas


# ── Persistencia ───────────────────────────────────────────────────────────────
def load_progress(json_path: Path) -> list:
    if json_path.exists():
        return json.loads(json_path.read_text())
    return []


def save_results(out_dir: Path, rows: list):
    csv_path  = out_dir / 'labels.csv'
    json_path = out_dir / 'labels.json'

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n  💾 Guardado: {len(rows)} etiquetas → {out_dir.name}/labels.csv")


def print_summary(rows, skipped):
    counts = {'FLUIDO': 0, 'DENSO': 0, 'COLAPSO': 0}
    for r in rows:
        counts[r['estado']] = counts.get(r['estado'], 0) + 1
    total = sum(counts.values())
    print(f"\n{'─'*40}")
    print(f"  Total etiquetados : {total}")
    print(f"  FLUIDO   : {counts['FLUIDO']:4d}  ({counts['FLUIDO']/max(total,1)*100:.1f}%)")
    print(f"  DENSO    : {counts['DENSO']:4d}  ({counts['DENSO']/max(total,1)*100:.1f}%)")
    print(f"  COLAPSO  : {counts['COLAPSO']:4d}  ({counts['COLAPSO']/max(total,1)*100:.1f}%)")
    print(f"  Saltados : {skipped}")
    print(f"{'─'*40}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fecha',     required=True, help='YYYYMMDD')
    p.add_argument('--desde',     type=int, default=1, help='Empezar desde el frame N')
    p.add_argument('--sin-drive', action='store_true')
    p.add_argument('--proyecto',  type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    global PROJECT_ROOT, DATASET_DIR
    if args.proyecto:
        PROJECT_ROOT = Path(args.proyecto).resolve()
        DATASET_DIR  = PROJECT_ROOT / 'dataset'

    day_dir    = DATASET_DIR / args.fecha
    frames_dir = day_dir / 'frames'
    yolo_dir   = day_dir / 'yolo'
    meta_dir   = day_dir / 'meta'
    for d in [frames_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Descarga desde Drive si procede
    if not args.sin_drive:
        service    = get_drive_service()
        folder_id  = get_folder_id(service, DRIVE_FOLDER)
        frame_list = download_day(service, folder_id, args.fecha, frames_dir, meta_dir)
    else:
        frame_list = sorted(frames_dir.glob('*.jpg'))
        if not frame_list:
            print(f"❌ No hay imágenes en {frames_dir}")
            sys.exit(1)
        print(f"[Local] {len(frame_list)} frames en {frames_dir}")

    # Detectar si existe carpeta yolo/ con imágenes anotadas
    usando_yolo = yolo_dir.exists() and len(list(yolo_dir.glob('*.jpg'))) > 0
    if usando_yolo:
        print(f"✅ Carpeta yolo/ detectada — se mostrarán imágenes con anotaciones YOLO")
    else:
        print(f"⚠️  Sin carpeta yolo/ — se mostrarán frames originales")
        print(f"   Puedes generarla con: python tools/yolo_preview.py --fecha {args.fecha}")

    total = len(frame_list)
    print(f"\n{'='*50}")
    print(f"  Día {args.fecha}  ·  {total} frames")
    print(f"  Modo: {'YOLO preview' if usando_yolo else 'frames originales'}")
    print(f"  [F] Fluido  [D] Denso  [C] Colapso  [Z] Deshacer  [S] Saltar  [Q] Salir")
    print(f"{'='*50}\n")

    json_path  = day_dir / 'labels.json'
    rows       = load_progress(json_path)
    done_names = {r['filename'] for r in rows}
    skipped    = 0

    WIN = 'Etiquetador — Rotonda Anchieta'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 860, 620)

    start = max(0, args.desde - 1)
    i     = start

    while i < total:
        img_path = frame_list[i]
        fname    = img_path.name

        if fname in done_names:
            i += 1
            continue

        # Cargar imagen: yolo/ si existe, si no frames/
        display_path = yolo_dir / fname if usando_yolo and (yolo_dir / fname).exists() else img_path
        img = cv2.imread(str(display_path))
        if img is None:
            # Fallback a frame original si la imagen yolo falla
            img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i+1}/{total}] {fname} — error de lectura, saltando")
            i += 1
            continue

        # Metadatos
        meta = {}
        meta_path = meta_dir / fname.replace('.jpg', '.json')
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass

        canvas = draw_frame(img, fname, i + 1, total, meta, usando_yolo)
        cv2.imshow(WIN, canvas)

        action = None
        while action is None:
            key = cv2.waitKey(0) & 0xFF
            action = KEY_MAP.get(key)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                save_results(day_dir, rows)
                print_summary(rows, skipped)
                return

        if action == 'QUIT':
            break

        elif action == 'SKIP':
            skipped += 1
            print(f"  [{i+1}/{total}] {fname} → SKIP")
            i += 1

        elif action == 'UNDO':
            if rows:
                undone = rows.pop()
                done_names.discard(undone['filename'])
                undone_path = frames_dir / undone['filename']
                if undone_path in frame_list:
                    i = frame_list.index(undone_path)
                else:
                    i = max(0, i - 1)
                print(f"  ↩  Deshecho: {undone['filename']} ({undone['estado']})")
            else:
                print("  ↩  Nada que deshacer")

        elif action in ESTADOS:
            confirm = draw_confirm(canvas.copy(), action)
            cv2.imshow(WIN, confirm)
            cv2.waitKey(280)

            # Siempre guarda el filename de frames/, no de yolo/
            row = {
                'filename':     fname,
                'timestamp':    meta.get('timestamp', fname.replace('.jpg', '')),
                'weekday':      meta.get('weekday', ''),
                'weekday_name': meta.get('weekday_name', ''),
                'hour':         meta.get('hour', ''),
                'minute':       meta.get('minute', ''),
                'is_rush_hour': meta.get('is_rush_hour', ''),
                'estado':       action,
            }
            rows.append(row)
            done_names.add(fname)
            print(f"  [{i+1}/{total}] {fname} → {action}")
            i += 1

            if len(rows) % 30 == 0:
                save_results(day_dir, rows)

    cv2.destroyAllWindows()
    save_results(day_dir, rows)
    print_summary(rows, skipped)


if __name__ == '__main__':
    main()
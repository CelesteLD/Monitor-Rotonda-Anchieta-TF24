"""
revisor_sospechosos.py — Revisión de etiquetas sospechosas
==========================================================
Muestra frames etiquetados como FLUIDO con un número alto de vehículos
para que puedas confirmar o corregir la etiqueta manualmente.

Uso:
    python tools/revisor_sospechosos.py
    python tools/revisor_sospechosos.py --min-vehicles 7
    python tools/revisor_sospechosos.py --min-vehicles 8 --fecha 20260512

Controles:
    F  → Confirmar como FLUIDO (no cambia nada)
    D  → Reetiqueta como DENSO
    C  → Reetiqueta como COLAPSO
    S  → Saltar sin cambios
    Q  → Guardar y salir

Al terminar guarda los cambios en labels.csv y labels.json de cada día afectado.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ─── Rutas ───────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"

COLOR_ESTADO = {
    "FLUIDO":  (0, 210, 0),
    "DENSO":   (0, 165, 255),
    "COLAPSO": (0, 0, 220),
}

# Tamaño fijo del canvas de visualización
CANVAS_W   = 900
IMG_H      = 540   # altura reservada para la imagen
PANEL_H    = 130   # altura del panel de información inferior
CANVAS_H   = IMG_H + PANEL_H


# ─── Argumentos ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Revisor de etiquetas sospechosas")
    p.add_argument("--min-vehicles", type=int, default=7,
                   help="Umbral mínimo de vehicle_count (default: 7)")
    p.add_argument("--fecha", type=str, default=None,
                   help="Revisar solo un día concreto (ej: 20260512)")
    return p.parse_args()


# ─── Carga de datos ───────────────────────────────────────────

def cargar_sospechosos(min_vehicles: int, fecha_filtro):
    sospechosos = []

    dias = sorted(d for d in DATASET_DIR.iterdir() if d.is_dir())
    if fecha_filtro:
        dias = [d for d in dias if d.name == fecha_filtro]
        if not dias:
            print(f"❌ No se encontró el día {fecha_filtro} en {DATASET_DIR}")
            sys.exit(1)

    for day_dir in dias:
        csv_path = day_dir / "labels.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "vehicle_count" not in df.columns:
            continue

        mascara   = (df["estado"] == "FLUIDO") & (df["vehicle_count"] >= min_vehicles)
        candidatos = df[mascara].copy()

        for _, row in candidatos.iterrows():
            img_path = day_dir / "frames" / row["filename"]
            if not img_path.exists():
                continue
            sospechosos.append({
                "fecha":         day_dir.name,
                "csv_path":      csv_path,
                "filename":      row["filename"],
                "img_path":      img_path,
                "hour":          row.get("hour", "?"),
                "vehicle_count": int(row["vehicle_count"]),
                "estado_actual": row["estado"],
                "estado_nuevo":  None,
            })

    return sospechosos


# ─── Renderizado ─────────────────────────────────────────────

def render_frame(img_orig: np.ndarray, item: dict, idx: int, total: int) -> np.ndarray:
    """
    Construye un canvas fijo CANVAS_W × CANVAS_H:
      - Parte superior: imagen centrada con letterbox negro
      - Parte inferior: panel de información con fondo oscuro
    """
    # ── Canvas negro base ──
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # ── Imagen con letterbox en la zona superior ──
    ih, iw = img_orig.shape[:2]
    scale  = min(CANVAS_W / iw, IMG_H / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img_orig, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = (CANVAS_W - nw) // 2
    y0 = (IMG_H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized

    # ── Panel inferior ──
    panel_y = IMG_H
    cv2.rectangle(canvas, (0, panel_y), (CANVAS_W, CANVAS_H), (25, 25, 25), -1)
    cv2.line(canvas, (0, panel_y), (CANVAS_W, panel_y), (60, 60, 60), 1)

    estado  = item["estado_actual"]
    color   = COLOR_ESTADO.get(estado, (200, 200, 200))
    vc      = item["vehicle_count"]
    fecha   = item["fecha"]
    fichero = item["filename"]
    hora    = item["hour"]

    # Fila 1 — progreso y archivo
    txt1 = f"[{idx+1}/{total}]   {fecha} / {fichero}   |   hora: {hora}"
    cv2.putText(canvas, txt1,
                (16, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # Fila 2 — etiqueta actual + conteo
    cv2.putText(canvas, "Etiqueta actual:",
                (16, panel_y + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, estado,
                (210, panel_y + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"   |   vehiculos detectados: {vc}",
                (310, panel_y + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 1, cv2.LINE_AA)

    # Fila 3 — controles
    controles = [
        ("[F]", "FLUIDO",  COLOR_ESTADO["FLUIDO"]),
        ("[D]", "DENSO",   COLOR_ESTADO["DENSO"]),
        ("[C]", "COLAPSO", COLOR_ESTADO["COLAPSO"]),
        ("[S]", "Saltar",  (140, 140, 140)),
        ("[Q]", "Guardar y salir", (140, 140, 140)),
    ]
    x = 16
    y = panel_y + 100
    for key, label, col in controles:
        cv2.putText(canvas, key,
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)
        x += cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0][0] + 4
        cv2.putText(canvas, f"{label}   ",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1, cv2.LINE_AA)
        x += cv2.getTextSize(f"{label}   ", cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]

    return canvas


# ─── Guardado ─────────────────────────────────────────────────

def guardar_cambios(sospechosos: list):
    cambios_por_csv: dict = {}
    for item in sospechosos:
        if item["estado_nuevo"] is None:
            continue
        csv_path = item["csv_path"]
        if csv_path not in cambios_por_csv:
            cambios_por_csv[csv_path] = {}
        cambios_por_csv[csv_path][item["filename"]] = item["estado_nuevo"]

    if not cambios_por_csv:
        print("\n✅ No hubo cambios que guardar.")
        return

    total_cambios = sum(len(v) for v in cambios_por_csv.values())
    print(f"\n💾 Guardando {total_cambios} cambio(s)...")

    for csv_path, mapa in cambios_por_csv.items():
        # ── labels.csv ──
        df = pd.read_csv(csv_path)
        mascara = df["filename"].isin(mapa)
        df.loc[mascara, "estado"] = df.loc[mascara, "filename"].map(mapa)
        df.to_csv(csv_path, index=False)
        print(f"   ✅ {csv_path.parent.name}/labels.csv  ({len(mapa)} filas)")

        # ── labels.json (si existe) ──
        json_path = csv_path.parent / "labels.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for fn, nuevo in mapa.items():
                        if fn in data:
                            data[fn] = nuevo
                elif isinstance(data, list):
                    for entry in data:
                        fn = entry.get("filename") or entry.get("image")
                        if fn in mapa:
                            entry["estado"] = mapa[fn]
                with open(json_path, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"   ✅ {csv_path.parent.name}/labels.json")
            except Exception as e:
                print(f"   ⚠️  No se pudo actualizar labels.json: {e}")


# ─── Bucle principal ──────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 58)
    print("  Revisor de etiquetas sospechosas")
    print(f"  Umbral: FLUIDO con >= {args.min_vehicles} vehículos")
    if args.fecha:
        print(f"  Filtrando día: {args.fecha}")
    print("=" * 58)

    sospechosos = cargar_sospechosos(args.min_vehicles, args.fecha)

    if not sospechosos:
        print(f"\n✅ No hay frames FLUIDO con >= {args.min_vehicles} vehículos.")
        return

    print(f"\n🔍 {len(sospechosos)} frames sospechosos encontrados.\n")

    cv2.namedWindow("Revisor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Revisor", CANVAS_W, CANVAS_H)

    confirmados   = 0
    reetiquetados = 0
    saltados      = 0

    for idx, item in enumerate(sospechosos):
        img = cv2.imread(str(item["img_path"]))
        if img is None:
            print(f"  ⚠️  No se pudo leer {item['img_path']}")
            continue

        while True:
            canvas = render_frame(img, item, idx, len(sospechosos))
            cv2.imshow("Revisor", canvas)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("f"), ord("F")):
                item["estado_nuevo"] = None
                confirmados += 1
                print(f"  [{idx+1}/{len(sospechosos)}] ✅ Confirmado FLUIDO  — {item['filename']}")
                break
            elif key in (ord("d"), ord("D")):
                item["estado_nuevo"] = "DENSO"
                reetiquetados += 1
                print(f"  [{idx+1}/{len(sospechosos)}] 🔄 FLUIDO → DENSO     — {item['filename']}")
                break
            elif key in (ord("c"), ord("C")):
                item["estado_nuevo"] = "COLAPSO"
                reetiquetados += 1
                print(f"  [{idx+1}/{len(sospechosos)}] 🔄 FLUIDO → COLAPSO   — {item['filename']}")
                break
            elif key in (ord("s"), ord("S")):
                item["estado_nuevo"] = None
                saltados += 1
                print(f"  [{idx+1}/{len(sospechosos)}] ⏭  Saltado            — {item['filename']}")
                break
            elif key in (ord("q"), ord("Q")):
                print(f"\n⚠️  Salida anticipada en frame {idx+1}/{len(sospechosos)}")
                guardar_cambios(sospechosos)
                cv2.destroyAllWindows()
                _resumen(confirmados, reetiquetados, saltados, len(sospechosos) - idx - 1)
                return

    cv2.destroyAllWindows()
    guardar_cambios(sospechosos)
    _resumen(confirmados, reetiquetados, saltados, 0)


def _resumen(confirmados, reetiquetados, saltados, pendientes):
    print("\n" + "=" * 58)
    print("  Resumen de la sesión")
    print(f"    Confirmados FLUIDO : {confirmados}")
    print(f"    Reetiquetados      : {reetiquetados}")
    print(f"    Saltados           : {saltados}")
    if pendientes:
        print(f"    Sin revisar        : {pendientes}")
    print("=" * 58)
    if reetiquetados:
        print("\n  ⚠️  Recuerda reentrenar el modelo:")
        print("  docker exec rotonda_spark spark-submit ...")


if __name__ == "__main__":
    main()
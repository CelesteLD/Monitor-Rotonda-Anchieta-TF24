#!/usr/bin/env python3
"""
dedup_frames.py — Elimina imágenes duplicadas por contenido (hash MD5)
Uso:
  python tools/dedup_frames.py --fecha 20260508
  python tools/dedup_frames.py --fecha 20260508 --dry-run   # solo muestra, no borra
"""

import hashlib
import argparse
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / 'backend').exists() else SCRIPT_DIR.parent
DATASET_DIR  = PROJECT_ROOT / 'dataset'


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def dedup(fecha: str, dry_run: bool):
    frames_dir = DATASET_DIR / fecha / 'frames'
    if not frames_dir.exists():
        print(f"❌ No existe {frames_dir}")
        return

    images = sorted(frames_dir.glob('*.jpg'))
    print(f"📂 {frames_dir}")
    print(f"   {len(images)} imágenes encontradas\n")

    seen    = {}   # hash → primer fichero encontrado
    to_delete = []

    for img in images:
        h = md5(img)
        if h in seen:
            to_delete.append(img)
            print(f"  DUP  {img.name}  ==  {seen[h].name}")
        else:
            seen[h] = img

    print(f"\n{'─'*50}")
    print(f"  Únicos   : {len(seen)}")
    print(f"  Duplicados: {len(to_delete)}")

    if not to_delete:
        print("  ✅ No hay duplicados.")
        return

    if dry_run:
        print("\n  [dry-run] No se ha borrado nada. Quita --dry-run para borrar.")
        return

    confirm = input(f"\n  ¿Borrar {len(to_delete)} duplicados? [s/N] ").strip().lower()
    if confirm != 's':
        print("  Cancelado.")
        return

    for f in to_delete:
        f.unlink()
        print(f"  🗑  {f.name}")

    print(f"\n  ✅ {len(to_delete)} duplicados eliminados. Quedan {len(seen)} imágenes.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fecha',   required=True, help='YYYYMMDD')
    p.add_argument('--dry-run', action='store_true', help='Solo muestra, no borra')
    args = p.parse_args()
    dedup(args.fecha, args.dry_run)
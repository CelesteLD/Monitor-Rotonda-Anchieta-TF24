"""
mask_selector.py — Define la máscara de carretera dibujando un polígono

Uso:
    python mask_selector.py                        # descarga frame en vivo
    python mask_selector.py --image imagen.jpg     # usa imagen local

Haz clic para añadir puntos al polígono.
Pulsa ENTER para cerrar y guardar.
Pulsa Z para deshacer el último punto.
Pulsa R para reiniciar.
"""

import cv2
import numpy as np
import warnings
import time
import json
import sys
import argparse

warnings.filterwarnings('ignore')

CAM_URL = 'https://cic.tenerife.es/e-Traffic3/data/camara-2701002-517.jpg'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer':    'https://cic.tenerife.es/web3/camaras_def/cam_web_2701002_517_def.html',
}

points = []
frame_display = None

def fetch_frame_live():
    import requests
    r = requests.get(CAM_URL, params={'t': int(time.time()*1000)},
                     headers=HEADERS, verify=False, timeout=15)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.resize(img, (640, 480))

def fetch_frame_local(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'No se pudo leer: {path}')
    return cv2.resize(img, (640, 480))

def draw(frame_base):
    display = frame_base.copy()

    if len(points) >= 3:
        overlay = display.copy()
        mask = np.zeros(display.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points)], 255)
        outside = cv2.bitwise_not(mask)
        display[outside == 255] = (display[outside == 255] * 0.35).astype(np.uint8)

    for i in range(1, len(points)):
        cv2.line(display, points[i-1], points[i], (0, 0, 255), 2)

    if len(points) >= 3:
        cv2.line(display, points[-1], points[0], (0, 0, 255), 1)

    for i, p in enumerate(points):
        cv2.circle(display, p, 5, (255, 255, 255), -1)
        cv2.circle(display, p, 5, (0, 0, 255), 1)

    cv2.rectangle(display, (0, 455), (640, 480), (30, 30, 30), -1)
    cv2.putText(display,
                f'Puntos: {len(points)}  |  ENTER=guardar  Z=deshacer  R=reiniciar',
                (8, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    return display

def mouse_callback(event, x, y, flags, param):
    global points, frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.imshow('Mask Selector - Rotonda Anchieta TF-24', draw(frame_display))

def main():
    global points, frame_display

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=None,
                        help='Ruta a imagen local (si no se especifica, descarga de la cámara)')
    args = parser.parse_args()

    if args.image:
        print(f'Cargando imagen local: {args.image}')
        try:
            frame_display = fetch_frame_local(args.image)
            print(f'Frame: {frame_display.shape[1]}x{frame_display.shape[0]}px')
        except Exception as e:
            print(f'Error: {e}')
            sys.exit(1)
    else:
        print('Descargando frame en vivo de la cámara...')
        try:
            frame_display = fetch_frame_live()
            print(f'Frame: {frame_display.shape[1]}x{frame_display.shape[0]}px')
        except Exception as e:
            print(f'Error descargando frame: {e}')
            frame_display = np.zeros((480, 640, 3), dtype=np.uint8)

    print('\nInstrucciones:')
    print('  - Clic izquierdo para añadir puntos al polígono')
    print('  - Sigue el borde de la carretera (incluyendo todas las salidas)')
    print('  - ENTER para guardar')
    print('  - Z para deshacer último punto')
    print('  - R para reiniciar\n')

    cv2.namedWindow('Mask Selector - Rotonda Anchieta TF-24')
    cv2.setMouseCallback('Mask Selector - Rotonda Anchieta TF-24', mouse_callback)
    cv2.imshow('Mask Selector - Rotonda Anchieta TF-24', draw(frame_display))

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == 13:  # ENTER
            if len(points) < 3:
                print('  Necesitas al menos 3 puntos')
                continue
            break

        elif key == ord('z') or key == ord('Z'):
            if points:
                points.pop()
                print(f'  Deshecho. Puntos restantes: {len(points)}')
                cv2.imshow('Mask Selector - Rotonda Anchieta TF-24', draw(frame_display))

        elif key == ord('r') or key == ord('R'):
            points = []
            print('  Reiniciado')
            cv2.imshow('Mask Selector - Rotonda Anchieta TF-24', draw(frame_display))

        elif key == 27:  # ESC
            print('Cancelado')
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    print('\n' + '='*60)
    print('Copia este bloque en backend/services/detector.py')
    print('Reemplaza el array ROAD_MASK_POLYGON existente')
    print('='*60)
    print(f'\nROAD_MASK_POLYGON = np.array({points}, dtype=np.int32)\n')

    with open('mask_polygon.json', 'w') as f:
        json.dump(points, f)
    print('Polígono guardado en mask_polygon.json')

    # Mostrar resultado final
    poly   = np.array(points)
    result = frame_display.copy()
    mask   = np.zeros(result.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    outside = cv2.bitwise_not(mask)
    result[outside == 255] = (result[outside == 255] * 0.2).astype(np.uint8)
    cv2.polylines(result, [poly], True, (0, 0, 255), 2)

    cv2.imshow('Mascara final - pulsa cualquier tecla', result)
    cv2.imwrite('mask_resultado.jpg', result)
    print('Imagen guardada en mask_resultado.jpg')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

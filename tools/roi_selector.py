import cv2
import requests
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')

CAM_URL = 'https://cic.tenerife.es/e-Traffic3/data/camara-2701002-517.jpg'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer':    'https://cic.tenerife.es/web3/camaras_def/cam_web_2701002_517_def.html',
}

ROI_NAMES = [
    'tramo1_rotonda',
    'tramo2_rotonda',
    'tramo3_rotonda',
]

ROI_COLORS = {
    'tramo1_rotonda':   (0,   200, 255),
    'tramo2_rotonda':   (0,   140, 255),
    'tramo3_rotonda':   (255, 140,   0),
}

ROI_LABELS = {
    'tramo1_rotonda':   'Tramo 1 Rotonda (izq)',
    'tramo2_rotonda':   'Tramo 2 Rotonda (centro)',
    'tramo3_rotonda':   'Tramo 3 Rotonda (derecha)',
}

def fetch_frame():
    r = requests.get(CAM_URL, params={'t': int(time.time()*1000)},
                     headers=HEADERS, verify=False, timeout=15)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.resize(img, (640, 480))

def select_rois(frame):
    rois = {}
    overlay = frame.copy()

    for name in ROI_NAMES:
        label = ROI_LABELS[name]
        color = ROI_COLORS[name]

        display = overlay.copy()
        for n, (x1, y1, x2, y2) in rois.items():
            c = ROI_COLORS[n]
            cv2.rectangle(display, (x1, y1), (x2, y2), c, 2)
            cv2.putText(display, ROI_LABELS[n], (x1+4, y1+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

        cv2.rectangle(display, (0, 455), (640, 480), (30, 30, 30), -1)
        cv2.putText(display,
                    f'Dibuja: {label}  |  ESPACIO=confirmar  C=repetir',
                    (8, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        print(f'\n  -> Dibuja ahora: {label}')

        r = cv2.selectROI(
            'ROI Selector - Rotonda Anchieta TF-24',
            display,
            fromCenter=False,
            showCrosshair=True
        )
        x, y, w, h = r
        if w == 0 or h == 0:
            print(f'  Zona vacia, saltando {name}')
            continue

        rois[name] = (x, y, x + w, y + h)
        print(f'  OK {label}: ({x}, {y}, {x+w}, {y+h})')

        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
        cv2.putText(overlay, label, (x+4, y+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.destroyAllWindows()
    return rois

def print_result(rois):
    print('\n' + '='*60)
    print('Copia este bloque en backend/services/detector.py')
    print('Reemplaza el diccionario ROIS existente')
    print('='*60)
    print('\nROIS = {')
    for name, coords in rois.items():
        print(f"    '{name}': {coords},")
    print('}')
    print()

def main():
    print('Descargando frame de la camara...')
    try:
        frame = fetch_frame()
        print(f'Frame descargado: {frame.shape[1]}x{frame.shape[0]}px')
    except Exception as e:
        print(f'Error: {e}')
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print('\nZonas a definir (7):')
    for i, name in enumerate(ROI_NAMES, 1):
        print(f'  {i}. {ROI_LABELS[name]}')

    print('\nInstrucciones:')
    print('  - Arrastra con el raton para dibujar cada zona')
    print('  - Pulsa ESPACIO o ENTER para confirmar')
    print('  - Pulsa C para repetir la zona actual\n')

    rois = select_rois(frame)

    if rois:
        print_result(rois)
        result = frame.copy()
        for name, (x1, y1, x2, y2) in rois.items():
            color = ROI_COLORS[name]
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result, ROI_LABELS[name], (x1+4, y1+16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow('ROIs finales - pulsa cualquier tecla para cerrar', result)
        cv2.imwrite('rois_resultado.jpg', result)
        print('Imagen guardada en rois_resultado.jpg')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
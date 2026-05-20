"""
camera.py — Descarga periódica de frames + caché del último resultado de detección
"""

import os
import time
import hashlib
import requests
import warnings
from datetime import datetime
from threading import Lock

warnings.filterwarnings('ignore')

CAM_URL     = os.environ.get('CAM_URL', 'https://cic.tenerife.es/e-Traffic3/data/camara-2701002-517.jpg')
CAM_REFERER = os.environ.get('CAM_REFERER', 'https://cic.tenerife.es/web3/camaras_def/cam_web_2701002_517_def.html')
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
    'Referer':    CAM_REFERER,
    'Accept':     'image/jpeg,image/*',
}

# Estado del frame actual (bytes crudos)
_frame_state = {
    'bytes':     None,
    'timestamp': None,
    'hash':      None,
    'error':     None,
}

# Caché del último resultado procesado (imagen anotada + conteos)
_detection_cache = {
    'annotated_jpg':  None,
    'vehicle_count':  None,
    'roi_counts':     {},
    'estado':         None,
    'timestamp':      None,
    'metodo':         None,
    'confianza':      None,
    'cpp_elapsed_ms': None,   
    'cpp_threads':    None,   
}

_frame_lock     = Lock()
_detection_lock = Lock()

def get_current_frame() -> dict:
    with _frame_lock:
        return dict(_frame_state)

def get_detection_cache() -> dict:
    """Retorna la última imagen anotada y conteos ya procesados"""
    with _detection_lock:
        return dict(_detection_cache)

def set_detection_cache(annotated_jpg, vehicle_count, roi_counts, estado, timestamp,
                        metodo=None, confianza=None,
                        cpp_elapsed_ms=None, cpp_threads=None):  
    with _detection_lock:
        _detection_cache['annotated_jpg']  = annotated_jpg
        _detection_cache['vehicle_count']  = vehicle_count
        _detection_cache['roi_counts']     = roi_counts
        _detection_cache['estado']         = estado
        _detection_cache['timestamp']      = timestamp
        _detection_cache['metodo']         = metodo
        _detection_cache['confianza']      = confianza
        _detection_cache['cpp_elapsed_ms'] = cpp_elapsed_ms   
        _detection_cache['cpp_threads']    = cpp_threads  

def fetch_frame() -> bool:
    """Descarga un frame nuevo. Retorna True si es diferente al anterior."""
    global _frame_state
    try:
        r = requests.get(
            CAM_URL,
            params={'t': int(time.time() * 1000)},
            headers=HEADERS,
            verify=False,
            timeout=15
        )
        r.raise_for_status()

        content      = r.content
        current_hash = hashlib.md5(content).hexdigest()

        with _frame_lock:
            if current_hash == _frame_state['hash']:
                return False
            _frame_state = {
                'bytes':     content,
                'timestamp': datetime.now().isoformat(),
                'hash':      current_hash,
                'error':     None,
            }
        return True

    except Exception as e:
        with _frame_lock:
            _frame_state['error'] = str(e)
        print(f'[Camera] Error al descargar frame: {e}')
        return False
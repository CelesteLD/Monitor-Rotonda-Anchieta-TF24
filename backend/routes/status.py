from flask import Blueprint, jsonify
from services.camera   import get_detection_cache
from services.database import get_stats_today
from datetime import datetime

from services.scheduler import get_last_pipeline_ts, FETCH_INTERVAL

status_bp = Blueprint('status', __name__)

@status_bp.route('/status')
def status():
    """Estado actual — siempre sincronizado con la imagen en /api/frame"""
    now     = datetime.now()
    is_rush = now.hour in range(7, 10) or \
              now.hour in range(13, 16) or \
              now.hour in range(17, 20)

    cache   = get_detection_cache()
    last_ts = get_last_pipeline_ts()
    next_in = max(0, FETCH_INTERVAL - int((now - last_ts).total_seconds())) \
              if last_ts else None

    if cache['vehicle_count'] is None:
        return jsonify({
            'timestamp':     None,
            'vehicle_count': None,
            'roi_counts':    {},
            'estado':        'FLUIDO',
            'confianza':     None,
            'metodo':        'umbral',
            'cuda_used':     False,
            'is_rush_hour':  is_rush,
            'ready':         False,
            'next_in':       next_in,
            'message':       'Esperando primer frame de la camara...',
        })

    return jsonify({
        'timestamp':      cache['timestamp'],
        'vehicle_count':  cache['vehicle_count'],
        'roi_counts':     cache['roi_counts'],
        'estado':         cache['estado'],
        'confianza':      1.0,
        'metodo':         cache.get('metodo', 'umbral'),
        'cuda_used':      False,
        'is_rush_hour':   is_rush,
        'ready':          True,
        'next_in':        next_in,
        'cpp_elapsed_ms': cache.get('cpp_elapsed_ms'),
        'cpp_threads':    cache.get('cpp_threads'),
    })

@status_bp.route('/stats')
def stats():
    return jsonify(get_stats_today())

@status_bp.route('/test-telegram')
def test_telegram():
    from services.telegram import send_alert
    try:
        send_alert('🟢 Test de notificación — Monitor Rotonda Anchieta TF-24')
        return jsonify({'ok': True, 'message': 'Mensaje enviado'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
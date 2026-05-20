from flask import Blueprint, jsonify, request
from services.database import get_history, get_history_range

history_bp = Blueprint('history', __name__)

@history_bp.route('/history')
def history():
    """Histórico de detecciones. ?minutes=60 (por defecto)"""
    minutes = request.args.get('minutes', 60, type=int)
    minutes = min(minutes, 1440)  # máximo 24h
    rows    = get_history(minutes)
    for r in rows:
        if hasattr(r['timestamp'], 'isoformat'):
            r['timestamp'] = r['timestamp'].isoformat()
    return jsonify(rows)

@history_bp.route('/history/range')
def history_range():
    """
    Histórico por rango de fechas.
    ?desde=2026-05-07&hasta=2026-05-08
    Devuelve máximo 5000 filas ordenadas por timestamp ASC.
    """
    desde = request.args.get('desde')  # YYYY-MM-DD
    hasta = request.args.get('hasta')  # YYYY-MM-DD
    if not desde or not hasta:
        return jsonify({'error': 'Parámetros desde y hasta requeridos'}), 400
    try:
        rows = get_history_range(desde, hasta)
        for r in rows:
            if hasattr(r['timestamp'], 'isoformat'):
                r['timestamp'] = r['timestamp'].isoformat()
        return jsonify(rows)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
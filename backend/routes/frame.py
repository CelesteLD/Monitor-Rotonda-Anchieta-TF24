from flask import Blueprint, send_file, jsonify
from services.camera import get_current_frame, get_detection_cache
from io import BytesIO

frame_bp = Blueprint('frame', __name__)

@frame_bp.route('/frame')
def frame():
    """
    Devuelve la imagen anotada del último ciclo de detección.
    Siempre corresponde exactamente al mismo frame que generó el conteo actual.
    """
    cache = get_detection_cache()
    if not cache['annotated_jpg']:
        return jsonify({'error': 'Sin frame todavía', 'ready': False}), 202

    return send_file(
        BytesIO(cache['annotated_jpg']),
        mimetype='image/jpeg',
        as_attachment=False,
    )

@frame_bp.route('/frame/raw')
def frame_raw():
    """Devuelve el frame sin anotar"""
    data = get_current_frame()
    if not data['bytes']:
        return jsonify({'error': 'Sin frame todavía', 'ready': False}), 202
    return send_file(BytesIO(data['bytes']), mimetype='image/jpeg')
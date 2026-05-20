"""
events.py — SSE endpoint para notificar al frontend cuando hay nueva detección.
El frontend se suscribe una vez y recibe un evento cada vez que el scheduler
finaliza un ciclo. No más polling desincronizado.
"""

import json
import time
import queue
from flask import Blueprint, Response, stream_with_context
from threading import Lock

events_bp = Blueprint('events', __name__)

# Cola de suscriptores activos
_subscribers: list[queue.Queue] = []
_sub_lock = Lock()


def _add_subscriber() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=5)
    with _sub_lock:
        _subscribers.append(q)
    return q


def _remove_subscriber(q: queue.Queue):
    with _sub_lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


def notify_detection(payload: dict):
    """
    Llamado por el scheduler tras cada ciclo exitoso.
    Envía el evento a todos los clientes conectados.
    """
    msg = json.dumps(payload)
    dead = []
    with _sub_lock:
        for q in _subscribers:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
    for q in dead:
        _remove_subscriber(q)


@events_bp.route('/events')
def sse():
    """
    Endpoint SSE. El cliente recibe:
      - Un heartbeat cada 20s para mantener la conexión viva.
      - Un evento 'detection' con vehicle_count, estado y timestamp
        cada vez que el backend termina un ciclo.
    """
    q = _add_subscriber()

    def generate():
        # Heartbeat inicial para confirmar conexión
        yield "event: connected\ndata: {}\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=20)
                    yield f"event: detection\ndata: {msg}\n\n"
                except queue.Empty:
                    # Heartbeat para evitar timeouts de proxy/nginx
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            _remove_subscriber(q)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':   'no-cache',
            'X-Accel-Buffering': 'no',   # deshabilita buffering en nginx
        },
    )

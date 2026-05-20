import os
import socket
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import time

from services.camera     import fetch_frame, get_current_frame, set_detection_cache
from services.detector   import detect
from services.classifier import predict
from services.database   import save_deteccion, get_last_estado
from services.telegram   import send_alert

_spark_server  = None
_spark_clients = []
_spark_lock    = threading.Lock()

FETCH_INTERVAL = int(os.environ.get('FETCH_INTERVAL', 45))   # bajado de 100 → 45s
SPARK_HOST     = os.environ.get('SPARK_HOST', 'spark')
SPARK_PORT     = int(os.environ.get('SPARK_PORT', 9999))

_last_pipeline_ts = None

def _start_spark_server():
    global _spark_server
    _spark_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    _spark_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _spark_server.bind(('0.0.0.0', SPARK_PORT))
    _spark_server.listen(5)
    print(f'[Scheduler] Servidor socket escuchando en :{SPARK_PORT}')

    def accept_loop():
        while True:
            try:
                conn, addr = _spark_server.accept()
                print(f'[Scheduler] Spark conectado desde {addr}')
                with _spark_lock:
                    _spark_clients.append(conn)
            except Exception:
                break

    threading.Thread(target=accept_loop, daemon=True).start()

def _send_to_spark(data: dict):
    msg = (json.dumps(data) + '\n').encode()
    with _spark_lock:
        dead = []
        for conn in _spark_clients:
            try:
                conn.sendall(msg)
            except Exception:
                dead.append(conn)
        for conn in dead:
            _spark_clients.remove(conn)

def pipeline():
    now = datetime.now()
    global _last_pipeline_ts
    timings = {}

    # 1. Descargar frame
    t0 = time.perf_counter()
    is_new = fetch_frame()
    timings['fetch_ms'] = (time.perf_counter() - t0) * 1000
    if not is_new:
        print(f'[{now:%H:%M:%S}] Frame sin cambios, saltando')
        return

    frame_data = get_current_frame()
    if not frame_data['bytes']:
        return

    # 2. Detectar vehículos (YOLO + C++ OpenMP)
    t0 = time.perf_counter()
    result = detect(frame_data['bytes'])
    timings['detect_ms'] = (time.perf_counter() - t0) * 1000

    vehicle_count = result['vehicle_count']
    roi_counts    = result['roi_counts']
    annotated_jpg = result['annotated_jpg']
    timings['cpp_ms'] = result.get('cpp_elapsed_ms', 0)

    # 3. Clasificar estado
    t0 = time.perf_counter()
    is_rush = now.hour in range(7, 10) or \
              now.hour in range(13, 16) or \
              now.hour in range(17, 20)
    classification = predict(vehicle_count, now.hour, is_rush, result.get('visual_features'))
    timings['classify_ms'] = (time.perf_counter() - t0) * 1000

    estado = classification['estado']

    # Log del desglose
    print(
        f'[{now:%H:%M:%S}] Vehículos: {vehicle_count} → {estado} | '
        f'fetch={timings["fetch_ms"]:.0f}ms '
        f'detect={timings["detect_ms"]:.0f}ms '
        f'cpp={timings["cpp_ms"]:.3f}ms '
        f'classify={timings["classify_ms"]:.0f}ms'
    )

    last_estado = get_last_estado(n=1)

    # 4. Guardar imagen anotada + conteos en caché (sincronizados)
    set_detection_cache(
        annotated_jpg = annotated_jpg,
        vehicle_count = vehicle_count,
        roi_counts    = roi_counts,
        estado        = estado,
        timestamp     = frame_data['timestamp'],
        metodo        = classification['metodo'],
        confianza     = classification['confianza'],
        cpp_elapsed_ms = result.get('cpp_elapsed_ms'),  
        cpp_threads    = result.get('cpp_threads'),       
    )

    # 5. Notificar a clientes SSE (importación diferida para evitar ciclo en arranque)
    try:
        from routes.events import notify_detection
        notify_detection({
            'timestamp':     frame_data['timestamp'],
            'vehicle_count': vehicle_count,
            'estado':        estado,
            'roi_counts':    roi_counts,
        })
    except Exception as e:
        print(f'[Scheduler] SSE notify error: {e}')

    # 6. Guardar en PostgreSQL
    save_deteccion({
        'weekday':       now.weekday(),
        'hour':          now.hour,
        'minute':        now.minute,
        'is_rush_hour':  is_rush,
        'vehicle_count': vehicle_count,
        'estado':        estado,
        'confianza':     classification['confianza'],
    })

    # 7. Alerta Telegram si nuevo colapso
    if estado == 'COLAPSO' and last_estado != 'COLAPSO':
        try:
            send_alert(
                f'🔴 COLAPSO detectado — Rotonda Anchieta TF-24\n'
                f'Vehículos: {vehicle_count} | {now:%H:%M}',
                image_bytes=annotated_jpg,   
            )
            print(f'[Telegram] Alerta enviada ({vehicle_count} vehículos)')
        except Exception as e:
            print(f'[Telegram] Error al enviar alerta: {e}')

    # 8. Enviar a Spark
    _send_to_spark({
        'timestamp':     frame_data['timestamp'],
        'vehicle_count': vehicle_count,
        'estado':        estado,
        'hour':          now.hour,
        'weekday':       now.weekday(),
        'is_rush_hour':  int(is_rush),
    })
    _last_pipeline_ts = datetime.now()

def get_last_pipeline_ts():
    return _last_pipeline_ts

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        pipeline,
        trigger='interval',
        seconds=FETCH_INTERVAL,
        id='pipeline',
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    _start_spark_server()
    print(f'[Scheduler] Pipeline iniciado cada {FETCH_INTERVAL}s')

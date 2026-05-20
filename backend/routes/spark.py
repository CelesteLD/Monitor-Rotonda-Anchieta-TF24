from flask import Blueprint, jsonify, request
from services.database import _conn
from datetime import datetime, timedelta

spark_bp = Blueprint('spark', __name__)


@spark_bp.route('/spark/tendencias')
def tendencias():
    """
    Devuelve las últimas N ventanas de Spark Streaming.
    ?ventanas=12  → últimas 12 ventanas (por defecto, ~6 minutos)
    ?minutos=30   → ventanas de los últimos 30 minutos
    """
    ventanas = request.args.get('ventanas', 12, type=int)
    minutos  = request.args.get('minutos',  None, type=int)

    try:
        with _conn() as conn:
            with conn.cursor() as cur:

                if minutos:
                    desde = datetime.utcnow() - timedelta(minutes=minutos)
                    cur.execute("""
                        SELECT timestamp, avg_vehicles, max_vehicles,
                               tendencia, estado_ventana, estado_predicho, n_samples
                        FROM spark_ventanas
                        WHERE timestamp >= %s
                        ORDER BY timestamp ASC
                    """, (desde,))
                else:
                    cur.execute("""
                        SELECT timestamp, avg_vehicles, max_vehicles,
                               tendencia, estado_ventana, estado_predicho, n_samples
                        FROM spark_ventanas
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (ventanas,))

                rows = cur.fetchall()

        cols = ['timestamp', 'avg_vehicles', 'max_vehicles',
                'tendencia', 'estado_ventana', 'estado_predicho', 'n_samples']

        result = []
        for row in rows:
            r = dict(zip(cols, row))
            if hasattr(r['timestamp'], 'isoformat'):
                r['timestamp'] = r['timestamp'].isoformat()
            result.append(r)

        # Si ordenamos por DESC para LIMIT, revertir para el frontend
        if not minutos:
            result = list(reversed(result))

        # Calcular predicción de próximos 10 minutos basada en tendencia
        prediccion_futura = None
        if result:
            ultimo        = result[-1]
            tendencia     = ultimo.get('tendencia', 'ESTABLE')
            estado_actual = ultimo.get('estado_ventana', 'FLUIDO')
            recientes     = result[-4:]
            avg_reciente  = sum(r['avg_vehicles'] for r in recientes) / max(len(recientes), 1)

            if tendencia == 'SUBIENDO':
                if estado_actual == 'FLUIDO' and avg_reciente > 3:
                    prediccion_futura = 'DENSO'
                elif estado_actual == 'DENSO':
                    prediccion_futura = 'COLAPSO'
                else:
                    prediccion_futura = estado_actual
            elif tendencia == 'BAJANDO':
                if estado_actual == 'COLAPSO':
                    prediccion_futura = 'DENSO'
                elif estado_actual == 'DENSO':
                    prediccion_futura = 'FLUIDO'
                else:
                    prediccion_futura = estado_actual
            else:
                prediccion_futura = estado_actual

        return jsonify({
            'ventanas':         result,
            'prediccion_10min': prediccion_futura,
            'total':            len(result),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://rotonda:rotonda123@postgres:5432/rotonda')

def _conn():
    return psycopg2.connect(DATABASE_URL)

def save_deteccion(data: dict):
    sql = """
        INSERT INTO detecciones
            (weekday, hour, minute, is_rush_hour, vehicle_count, estado, confianza)
        VALUES
            (%(weekday)s, %(hour)s, %(minute)s, %(is_rush_hour)s,
             %(vehicle_count)s, %(estado)s, %(confianza)s)
    """
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, data)

def get_last_estado(n: int = 1) -> str | None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT estado FROM detecciones ORDER BY timestamp DESC LIMIT %s',
                (n,)
            )
            rows = cur.fetchall()
            if not rows:
                return None
            return rows[0][0]

def get_history(minutes: int = 60) -> list:
    sql = """
        SELECT timestamp, vehicle_count, estado, confianza
        FROM detecciones
        WHERE timestamp > NOW() - INTERVAL '%s minutes'
        ORDER BY timestamp ASC
    """
    with _conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (minutes,))
            return [dict(r) for r in cur.fetchall()]

def get_history_range(desde: str, hasta: str) -> list:
    """
    Devuelve detecciones entre desde (inclusive) y hasta (inclusive, fin del día).
    Máximo 5000 filas para no saturar el frontend.
    """
    sql = """
        SELECT timestamp, vehicle_count, estado, confianza, hour, weekday
        FROM detecciones
        WHERE timestamp >= %(desde)s::date
          AND timestamp <  %(hasta)s::date + INTERVAL '1 day'
        ORDER BY timestamp ASC
        LIMIT 5000
    """
    with _conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, {'desde': desde, 'hasta': hasta})
            return [dict(r) for r in cur.fetchall()]

def get_stats_today() -> dict:
    sql = """
        SELECT
            COUNT(*)                                        AS total,
            AVG(vehicle_count)                              AS avg_vehicles,
            MAX(vehicle_count)                              AS max_vehicles,
            SUM(CASE WHEN estado = 'COLAPSO' THEN 1 END)   AS colapsos,
            SUM(CASE WHEN estado = 'DENSO'   THEN 1 END)   AS densos,
            SUM(CASE WHEN estado = 'FLUIDO'  THEN 1 END)   AS fluidos
        FROM detecciones
        WHERE timestamp::date = CURRENT_DATE
    """
    with _conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return dict(cur.fetchone())
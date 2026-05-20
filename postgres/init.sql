CREATE TABLE IF NOT EXISTS detecciones (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    weekday     SMALLINT    NOT NULL,
    hour        SMALLINT    NOT NULL,
    minute      SMALLINT    NOT NULL,
    is_rush_hour BOOLEAN    NOT NULL,
    vehicle_count INTEGER   NOT NULL,
    estado      VARCHAR(10) NOT NULL,  -- FLUIDO | DENSO | COLAPSO
    confianza   FLOAT,
    image_path  TEXT
);

CREATE INDEX idx_detecciones_timestamp ON detecciones(timestamp DESC);
CREATE INDEX idx_detecciones_estado    ON detecciones(estado);

CREATE TABLE IF NOT EXISTS spark_ventanas (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    avg_vehicles    FLOAT       NOT NULL,
    max_vehicles    INTEGER     NOT NULL,
    tendencia       VARCHAR(10) NOT NULL,
    estado_ventana  VARCHAR(10) NOT NULL,
    estado_predicho VARCHAR(20),                -- predicción MLlib (NULL si no hay modelo)
    n_samples       INTEGER     NOT NULL
);

CREATE INDEX idx_spark_ventanas_ts ON spark_ventanas(timestamp DESC);

CREATE TABLE IF NOT EXISTS alertas (
    id          SERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tipo        VARCHAR(20) NOT NULL,  -- COLAPSO_INICIO | COLAPSO_FIN
    duracion_min INTEGER,
    notificado  BOOLEAN DEFAULT FALSE
);
# Pipeline de Machine Learning — Rotonda Anchieta TF-24

Sistema de clasificación del estado del tráfico en la rotonda de la Autopista del Norte (TF-24) mediante visión por computador y aprendizaje automático distribuido con Apache Spark MLlib.

---

## Índice

- [Arquitectura del pipeline](#arquitectura-del-pipeline)
- [Scripts del pipeline](#scripts-del-pipeline)
- [Geometría y features](#geometría-y-features)
- [Comandos de ejecución](#comandos-de-ejecución)
- [Resultados del modelo](#resultados-del-modelo)
- [Evolución del modelo](#evolución-del-modelo)

---

## Arquitectura del pipeline

```
Google Drive
     │
     ▼
etiquetador_manual.py          →  dataset/202605XX/frames/
                                  dataset/202605XX/meta/
     │
     ▼
yolo_preview.py                →  dataset/202605XX/yolo/         (imágenes anotadas)
                                  dataset/202605XX/yolo/counts.json
     │
     ▼
etiquetador_manual.py          →  dataset/202605XX/labels.csv
  (--sin-drive, modo YOLO)        dataset/202605XX/labels.json
     │
     ▼
enrich_vehicle_count.py        →  labels.csv + columna vehicle_count
  (lee yolo/counts.json)
     │
     ▼
train_spark.py                 →  /data/models/traffic_model.joblib
  (Spark MLlib, CrossValidator)   /data/models/label_encoder.joblib
                                  /data/models/feature_names.json
     │
     ▼
docker compose restart backend →  modelo en producción
```

### Estructura de directorios del dataset

```
dataset/
└── 20260507/
    ├── frames/          ← imágenes originales descargadas de Drive
    ├── meta/            ← JSONs con metadatos de cada frame (hora, día, etc.)
    ├── yolo/            ← imágenes con bboxes YOLO anotados + counts.json
    ├── labels.csv       ← etiquetas + vehicle_count (ground truth)
    └── labels.json      ← mismo contenido en JSON
```

---

## Scripts del pipeline

### `tools/yolo_preview.py`

Ejecuta YOLOv8 sobre todos los frames de un día y genera imágenes anotadas en `dataset/202605XX/yolo/`. Cada imagen muestra:

- Contorno verde de la máscara de la rotonda
- Rectángulos de cada ROI (tramo1, tramo2, tramo3) en colores distintos
- Bounding boxes de vehículos detectados (verde = dentro de ROI, gris = fuera)
- Contador total de vehículos en la esquina superior izquierda

También genera `yolo/counts.json` con el conteo por filename para que `enrich_vehicle_count.py` no tenga que re-ejecutar YOLO.

```bash
python3 tools/yolo_preview.py --fecha 20260507
python3 tools/yolo_preview.py --fecha 20260507 --modelo /ruta/yolov8s.pt
```

---

### `tools/etiquetador_manual.py`

Herramienta de etiquetado manual frame a frame. Si existe `yolo/` para el día, muestra automáticamente las imágenes anotadas en vez de los frames originales, lo que permite etiquetar con información objetiva del conteo de vehículos.

El CSV siempre guarda el filename original de `frames/`, independientemente de qué imagen se esté mostrando.

| Tecla | Acción |
|-------|--------|
| `F` | Etiquetar como FLUIDO |
| `D` | Etiquetar como DENSO |
| `C` | Etiquetar como COLAPSO |
| `Z` | Deshacer última etiqueta |
| `S` | Saltar frame |
| `Q` | Guardar y salir |

```bash
# Descarga frames desde Drive y etiqueta
python tools/etiquetador_manual.py --fecha 20260507

# Solo etiqueta (frames ya descargados), usando yolo/ si existe
python tools/etiquetador_manual.py --fecha 20260507 --sin-drive

# Reanudar desde el frame 80
python tools/etiquetador_manual.py --fecha 20260507 --sin-drive --desde 80
```

---

### `tools/revisor_sospechosos.py`

Herramienta de revisión de etiquetas potencialmente incorrectas. Filtra frames etiquetados como FLUIDO con un número alto de vehículos detectados y permite confirmar o corregir la etiqueta visualmente.

```bash
# Revisar todos los días (umbral por defecto: >=7 vehículos)
python tools/revisor_sospechosos.py

# Solo un día concreto
python tools/revisor_sospechosos.py --fecha 20260512

# Subir el umbral para ser más selectivo
python tools/revisor_sospechosos.py --min-vehicles 9
```

---

### `backend/enrich_vehicle_count.py`

Añade la columna `vehicle_count` a los `labels.csv` de cada día. Lee los conteos pre-calculados de `yolo/counts.json` si existen (modo rápido), o ejecuta YOLO directamente como fallback para días que no los tengan.

Se ejecuta dentro del contenedor `rotonda_backend` porque necesita acceso al modelo YOLO y al dataset montado.

```bash
docker exec rotonda_backend python /app/enrich_vehicle_count.py
```

---

### `spark/train_spark.py`

Script principal de entrenamiento. Ejecuta el pipeline completo de ML:

1. Lee todos los `labels.csv` del dataset
2. Extrae 15 features visuales por imagen con OpenCV vía `mapPartitions` (distribuido)
3. Calcula 7 features de metadatos incluyendo `vehicle_count`, `congestion_hour_score` e `is_peak_hour`
4. Balancea clases con `sample_weight` (inversamente proporcional a frecuencia)
5. Entrena un Pipeline MLlib: `StringIndexer → VectorAssembler → RandomForestClassifier`
6. Optimiza hiperparámetros con `CrossValidator` 5-fold
7. Exporta el modelo final a `/data/models/` en formato sklearn `.joblib`

```bash
docker exec rotonda_spark \
  /opt/spark/bin/spark-submit \
    --master local[*] \
    --driver-memory 6g \
    --conf spark.executor.memory=6g \
    --conf spark.memory.fraction=0.8 \
    --conf spark.sql.shuffle.partitions=4 \
    /opt/spark_jobs/train_spark.py \
  2>/dev/null | tee /tmp/train_output.txt
```

---

## Geometría y features

### Máscara y ROIs

El sistema trabaja sobre una imagen normalizada a 640×480px. La máscara poligonal delimita el interior de la rotonda, descartando todo lo externo. Dentro de la máscara se definen tres ROIs:

| ROI | Coordenadas |
|-----|-------------|
| `tramo1_rotonda` | (413, 246) → (640, 404) |
| `tramo2_rotonda` | (184, 200) → (409, 337) |
| `tramo3_rotonda` | (3, 158) → (179, 296) |

Un vehículo se asigna a un ROI si su bounding box solapa al menos el **50%** con ese ROI (`OVERLAP_THRESHOLD = 0.5`). Cada vehículo se cuenta una sola vez aunque solape con varios ROIs.

### Features del modelo (22 en total)

**Features de metadatos (7):**

| Feature | Descripción |
|---------|-------------|
| `hour` | Hora del frame (0-23) |
| `weekday` | Día de la semana (0=lunes) |
| `minute` | Minuto del frame |
| `is_rush_hour` | Flag hora punta (calculado en captura) |
| `vehicle_count` | Número de vehículos dentro de los ROIs (YOLO) |
| `congestion_hour_score` | % histórico de congestión para esa hora |
| `is_peak_hour` | 1 si 11h ≤ hora ≤ 16h (franja de mayor congestión) |

El `congestion_hour_score` se calculó sobre el dataset completo:

| Hora | % Congestión |
|------|-------------|
| 8h | 20.3% |
| 9h | 16.3% |
| 10h | 23.8% |
| 11h | 32.2% |
| 12h | 43.7% |
| 13h | 46.0% |
| 14h | 36.5% |
| 15h | 43.3% |
| 16h | 42.0% |
| 17h | 21.9% |
| 18h | 9.8% |
| 19h | 5.6% |

**Features visuales (15) — 5 por cada ROI:**

| Feature | Descripción |
|---------|-------------|
| `brightness_mean` | Media de brillo en canal V (HSV) |
| `brightness_std` | Desviación estándar del brillo |
| `saturation_mean` | Media de saturación en canal S (HSV) |
| `edge_density` | Densidad de bordes Canny normalizada |
| `gray_contrast` | Contraste en escala de grises (std) |

---

## Comandos de ejecución

### Pipeline completo para un día nuevo

```bash
# 1. Descargar frames desde Google Drive
python tools/etiquetador_manual.py --fecha 20260520

# 2. Generar imágenes anotadas con YOLO
python3 tools/yolo_preview.py --fecha 20260520

# 3. Etiquetar viendo las detecciones YOLO
python tools/etiquetador_manual.py --fecha 20260520 --sin-drive

# 4. Añadir vehicle_count al CSV
docker exec rotonda_backend python /app/enrich_vehicle_count.py

# 5. Reentrenar el modelo
docker exec rotonda_spark \
  /opt/spark/bin/spark-submit \
    --master local[*] \
    --driver-memory 6g \
    --conf spark.executor.memory=6g \
    --conf spark.memory.fraction=0.8 \
    --conf spark.sql.shuffle.partitions=4 \
    /opt/spark_jobs/train_spark.py \
  2>/dev/null | tee /tmp/train_output.txt

# 6. Aplicar el modelo en producción
docker compose restart backend
```

### Revisar etiquetas sospechosas tras un reetiquetado

```bash
python tools/revisor_sospechosos.py --min-vehicles 9
```

---

## Resultados del modelo

Entrenamiento final con 7 días de datos (2592 imágenes, 22 features):

```
==========================================================
  Entrenamiento MLlib — Rotonda Anchieta TF-24
  3 ROIs · 22 features · RandomForest + CrossValidator
==========================================================

   Distribución de clases:
   FLUIDO :  1326  (51.2%)
   DENSO  :   989  (38.2%)
   COLAPSO:   277  (10.7%)

   Split temporal → train: 2204  test: 388 (20260515)

   Mejor combinación:
     numTrees = 200
     maxDepth = 12
   CV accuracy por fold:
     fold 1: 0.8541
     fold 2: 0.8553
     fold 3: 0.8592
     fold 4: 0.8593

   Accuracy : 0.9021 (90.2%)
   weightedPrecision   : 0.9011
   weightedRecall      : 0.9021
   f1                  : 0.9008

   Matriz de confusión:
   +-------+----------+-----+
   | label | pred     |  n  |
   +-------+----------+-----+
   |FLUIDO | FLUIDO   | 193 |
   |FLUIDO | DENSO    |   7 |
   |DENSO  | FLUIDO   |  18 |
   |DENSO  | DENSO    | 132 |
   |DENSO  | COLAPSO  |   5 |
   |COLAPSO| DENSO    |   8 |
   |COLAPSO| COLAPSO  |  25 |
   +-------+----------+-----+

   Top 10 features más importantes:
     vehicle_count              0.4445  ████████████████████████████
     tramo3_rotonda_edge_density 0.0635  ████
     tramo2_rotonda_edge_density 0.0504  ███
     tramo2_rotonda_saturation   0.0410  ███
     tramo2_rotonda_brightness   0.0365  ██
     tramo3_rotonda_saturation   0.0335  ██
     congestion_hour_score       0.0278  ██
     tramo2_gray_contrast        0.0262  ██
     tramo2_brightness_std       0.0257  ██
     minute                      0.0255  ██
```

---

## Evolución del modelo

La mejora más significativa no vino de cambios en el modelo sino de la **calidad del etiquetado**:

| Versión | Cambio | Accuracy | F1 |
|---------|--------|----------|----|
| v1 | Dataset original | 76.5% | 0.721 |
| v2 | + class weights balanceados | 75.8% | 0.713 |
| v3 | + `congestion_hour_score` e `is_peak_hour` | 76.3% | 0.723 |
| v4 | + reetiquetado parcial con revisor | 80.4% | 0.787 |
| v5 | + pipeline YOLO preview + reetiquetado completo | **90.2%** | **0.901** |

La lección principal: etiquetar viendo las detecciones de YOLO eliminó la ambigüedad en la frontera FLUIDO/DENSO, que era el error más frecuente del modelo (62 de 85 casos DENSO clasificados como FLUIDO en v1, frente a 18 de 155 en v5).
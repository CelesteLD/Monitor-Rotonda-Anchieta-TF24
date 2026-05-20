# Herramientas — `tools/`

Tres scripts de calibración y etiquetado. Todos se ejecutan desde la **raíz del proyecto** con `py` (Windows).

---

## Requisitos previos

```powershell
py -m pip install opencv-python numpy requests google-api-python-client google-auth-oauthlib
```

El archivo `token.json` (credenciales de Google Drive) debe estar en:
```
C:\Users\<tu_usuario>\token.json
```

---

## 1. `mask_selector.py` — Dibuja la máscara de carretera

Define qué zona de la imagen es "carretera". Todo lo que quede fuera de este polígono es ignorado por el detector.

**Descargando el frame actual de la cámara:**
```powershell
py tools\mask_selector.py
```

**Usando una imagen local:**
```powershell
py tools\mask_selector.py --image backend\images\20260507_100859.jpg
```

**Controles:**
| Acción | Control |
|---|---|
| Añadir punto al polígono | Clic izquierdo |
| Deshacer último punto | `Z` |
| Reiniciar todo | `R` |
| Guardar y cerrar | `ENTER` |
| Cancelar | `ESC` |

**Salida:**
- `mask_polygon.json` — coordenadas del polígono (en la raíz del proyecto)
- `mask_resultado.jpg` — previsualización de la máscara
- Imprime en consola el bloque `ROAD_MASK_POLYGON = np.array(...)` listo para copiar en `backend\services\detector.py`

---

## 2. `roi_selector.py` — Define las zonas de conteo (ROIs)

Define los 6 rectángulos donde YOLO contará vehículos: 4 tramos de la rotonda y 2 salidas.

```powershell
py tools\roi_selector.py
```

Descarga automáticamente el frame actual de la cámara. No admite imagen local.

**Controles:**
| Acción | Control |
|---|---|
| Dibujar rectángulo | Arrastrar con el ratón |
| Confirmar zona | `ESPACIO` o `ENTER` |
| Repetir zona actual | `C` |

Se definen las zonas en este orden:
1. Tramo 1 Rotonda
2. Tramo 2 Rotonda
3. Tramo 3 Rotonda
4. Tramo 4 Rotonda
5. Salida La Esperanza
6. Salida Las Dominicas

**Salida:**
- `rois_resultado.jpg` — previsualización de las ROIs
- Imprime en consola el bloque `ROIS = { ... }` listo para copiar en `backend\services\detector.py`

---

## 3. `etiquetador_manual.py` — Etiqueta frames para el dataset de ML

Descarga los frames de un día desde Google Drive y te permite asignar manualmente el estado de tráfico a cada imagen.

**Etiquetar un día completo (descarga desde Drive):**
```powershell
py tools\etiquetador_manual.py --fecha 20260507
```

**Reanudar una sesión interrumpida desde el frame 80:**
```powershell
py tools\etiquetador_manual.py --fecha 20260507 --desde 80
```

**Si ya tienes los frames descargados localmente:**
```powershell
py tools\etiquetador_manual.py --fecha 20260507 --sin-drive
```

**Controles durante el etiquetado:**
| Acción | Tecla |
|---|---|
| FLUIDO | `F` |
| DENSO | `D` |
| COLAPSO | `C` |
| Deshacer última etiqueta | `Z` |
| Saltar frame | `S` |
| Guardar y salir | `Q` |

El progreso se guarda automáticamente cada 30 frames. Si cierras la ventana o pulsas `Q`, puedes reanudar con `--desde N`.

**Salida** en `dataset\<YYYYMMDD>\`:
```
dataset\
└── 20260507\
    ├── frames\          ← imágenes originales descargadas de Drive
    ├── meta\            ← metadatos de hora y día de cada frame
    ├── labels.csv       ← ground truth (filename, hora, estado)
    └── labels.json      ← mismo contenido en JSON
```

```
mv tools/dataset/* dataset/

ls dataset/
# debe mostrar: 20260507  20260508  20260511 20260512

docker exec rotonda_spark \
  /opt/spark/bin/spark-submit \
    --master local[*] \
    /opt/spark_jobs/train_spark.py
```

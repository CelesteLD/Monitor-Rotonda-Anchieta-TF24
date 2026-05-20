# ─────────────────────────────────────────────────────────────
#  spark/Dockerfile
#  Base: apache/spark:3.5.1-python3 (Python 3.8 interno)
#  Añade: OpenCV + scikit-learn + joblib para train_spark.py
#  Los scripts entran por bind mount (./spark:/opt/spark_jobs)
# ─────────────────────────────────────────────────────────────
FROM apache/spark:3.5.1-python3

USER root

# Dependencias del sistema para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python (compatibles con Python 3.8)
RUN pip install --no-cache-dir \
        opencv-python-headless==4.8.1.78 \
        scikit-learn==1.3.2 \
        numpy==1.24.4 \
        pandas==2.0.3 \
        joblib==1.3.2 \
        psycopg2-binary==2.9.9

USER spark
ARG BASE=python:3.11-slim-bookworm
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV ROI_COUNTER_BIN=/app/cpp/roi_counter

RUN apt-get update && apt-get install -y \
    python3-dev \
    libopencv-dev \
    libpq-dev \
    gcc g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ls -la /app/cpp/ && \
    g++ -O2 -fopenmp -o /app/cpp/roi_counter /app/cpp/roi_counter.cpp && \
    echo "roi_counter compilado OK" && \
    ls -la /app/cpp/roi_counter

EXPOSE 5000

CMD ["gunicorn", \
     "--worker-class", "gevent", \
     "--workers", "1", \
     "--worker-connections", "100", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "0", \
     "app:app"]
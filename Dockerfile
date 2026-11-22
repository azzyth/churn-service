# Versi Python sengaja disamakan dengan venv pengembangan (3.14) supaya
# artefak joblib/scikit-learn tidak mismatch saat dimuat di container.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
# libgomp1 wajib untuk LightGBM (slim image tidak menyertakannya).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY config.yaml ./
COPY src ./src
COPY app ./app
COPY models ./models

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p data/raw data/splits reports/figures logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

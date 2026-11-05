FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SENTINEL_HOME=/var/lib/sentinel
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock && useradd --create-home sentinel && mkdir -p /var/lib/sentinel && chown sentinel:sentinel /var/lib/sentinel
COPY sentinelml ./sentinelml
USER sentinel
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "sentinelml.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

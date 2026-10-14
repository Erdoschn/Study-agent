FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential default-jdk-headless nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "pytest>=8,<9" \
    && mkdir -p /sandbox /build \
    && chown -R 65534:65534 /sandbox /build

COPY sandbox_runner.py /opt/coder_runner.py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

USER 65534:65534
ENTRYPOINT ["python", "/opt/coder_runner.py"]

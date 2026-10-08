FROM python:3.13-slim

RUN pip install --no-cache-dir "pytest>=8,<9" \
    && mkdir -p /sandbox /tmp/home \
    && chown -R 65534:65534 /sandbox /tmp/home

COPY sandbox_runner.py /opt/coder_runner.py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp/home

USER 65534:65534
ENTRYPOINT ["python", "/opt/coder_runner.py"]

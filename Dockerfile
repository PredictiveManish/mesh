FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY mesh ./mesh
COPY dashboard ./dashboard

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

ENV MESH_DB=/data/mesh_fleet.db
VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/api/v1/health')" || exit 1

CMD ["mesh", "serve", "--host", "0.0.0.0", "--port", "8080", "--db", "/data/mesh_fleet.db"]

# syntax=docker/dockerfile:1.7
# ---- build: resolve deps into a venv once, cached on pyproject changes only ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# ---- api runtime: slim, non-root, no compilers ----
FROM python:3.12-slim AS api
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    COLDCHAIN_ROOT=/app
RUN useradd --create-home --uid 10001 app
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY alembic.ini ./
COPY migrations ./migrations
COPY data/policy ./data/policy
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"
CMD ["coldchain", "serve", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ---- ui runtime: talks to the api only, ships no app code or secrets ----
FROM python:3.12-slim AS ui
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 app && pip install "streamlit>=1.38" "httpx>=0.27" "pandas>=2.2"
WORKDIR /app
COPY ui/app.py ./app.py
USER app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]

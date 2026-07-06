# AgriShield Universal -- single-service production image.
# Serves the FastAPI backend AND the static dashboard from one process,
# so this one image is the entire deployment (Render Web Service /
# Railway service, both build straight from this Dockerfile).

FROM python:3.12-slim

WORKDIR /app

# opencv-python-headless avoids GUI deps, but numpy/opencv still need
# these two small shared libs present on a slim base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY static/ ./static/

# Render/Railway inject $PORT at runtime; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# Single worker is fine to start -- OpenCV work is CPU-bound per request and
# most PaaS free/hobby tiers give one shared vCPU. Bump --workers once you've
# sized the dyno/instance, or put a process manager (gunicorn) in front if
# you need multi-core concurrency.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]

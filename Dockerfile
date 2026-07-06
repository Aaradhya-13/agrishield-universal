FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files directly from your flat layout
COPY main.py .
COPY pipeline.py .
COPY classifier.py .
COPY index.html .

# Tell Render what the default port is, but read the dynamic variable at runtime
EXPOSE 10000

# Start command pointing straight to your flat main.py layout with dynamic port matching
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]

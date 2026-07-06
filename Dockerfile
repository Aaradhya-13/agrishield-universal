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

# Expose port
EXPOSE 8000

# Start command pointing straight to your flat main.py layout
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

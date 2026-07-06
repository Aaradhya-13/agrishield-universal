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

# Copy all your application files from the root directly into the container
COPY main.py .
COPY pipeline.py .
COPY classifier.py .
COPY index.html .

# Expose the dynamic port
EXPOSE 8000

# Start command (Now pointing straight to main:app in the root)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

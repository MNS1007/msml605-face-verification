FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by Pillow / torch
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libjpeg62-turbo-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY tests/ tests/
COPY README.md .

# Pre-download the FaceNet model weights so they are baked into the image
RUN python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')"

# Default entrypoint: CLI inference
ENTRYPOINT ["python", "scripts/infer.py"]

# Docker will install a slim linux OS with python 3.10 on it.
FROM python:3.10-slim

# Install system dependencies for OpenCV and torchreid
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Tells docker to work within the folder app that will be created
WORKDIR /app

# Copies the requirements.txt file to the app new folder (.)
COPY requirements.txt .
# Downloads the python dependencies in the requirements.txt we copied in the command above
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# How to build docker image:
#   docker build -t tagyourmoves-player-signature .

# How to run docker container (CPU):
#   docker run --rm -p 8004:8000 -v "${PWD}:/app" -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 tagyourmoves-player-signature uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# How to run docker container (GPU):
#   docker run --rm --gpus all -p 8004:8000 -v "${PWD}:/app" -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 tagyourmoves-player-signature uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# For Afeka machine:
#   sudo docker run --rm --runtime=nvidia --gpus all -p 8004:8000 -v "${PWD}:/app" -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1 tagyourmoves-player-signature uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
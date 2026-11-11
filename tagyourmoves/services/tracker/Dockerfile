FROM python:3.10-slim

# Install dependencies for OpenCV (required for Camera Motion Compensation)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]


# How to build docker image:
#   docker build -t tagyourmoves-tracker .

# How to run docker container:
#   docker run --rm -p 8002:8000 -v "${PWD}:/app" -e PYTHONPATH=/app tagyourmoves-tracker uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
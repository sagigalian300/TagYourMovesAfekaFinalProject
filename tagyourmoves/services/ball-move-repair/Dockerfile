# Docker will install a slim linux OS with python 3.10 on it.
FROM python:3.10-slim

# No system packages are installed: the service works on the frames JSON alone (no video, no
# OpenCV), so plain Python packages are all it needs, which keeps the image small.

# Tells docker to work within the folder app that will be created
WORKDIR /app

# Copies the requirements.txt file to the app new folder (.)
COPY requirements.txt .
# Downloads the python dependencies in the requirements.txt we copied in the command above
RUN pip install --no-cache-dir -r requirements.txt

# Copies the service code (main.py and src/); videos, weights and caches are excluded by .dockerignore
COPY . .

# The FastAPI server listens on port 8000
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# How to build docker image:
# docker will create an image named "tagyourmoves-ball-move-repair" and tells that the '.' (current) directory is the working space for the image
    # docker build -t tagyourmoves-ball-move-repair .

# How to run docker container:
# docker will create a copy of the image (container) that will be removed automatically afterwards, and maps port 8000 of the container to port 8000 of this computer.
    # docker run --rm -p 8000:8000 tagyourmoves-ball-move-repair
# then the API docs are at http://localhost:8000/docs

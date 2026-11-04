# Docker will install a slim linux OS with python 3.10 on it.
FROM python:3.10-slim

# RUN command will run the followed command in the VM terminal (in this example it is the dependencies for the OpenCV library, and clean right after)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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

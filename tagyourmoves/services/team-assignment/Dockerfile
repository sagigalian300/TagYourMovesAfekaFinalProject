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

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# How to build docker image:
# docker will create an image named "tagyourmoves-team-assignment" and tells that the '.' (current) directory is the working space for the image
    # docker build -t tagyourmoves-team-assignment .

# How to run docker container:
# docker will create a copy of the image (container) that will be removed automatically afterwards, docker will create a volume (two side tunnel between current dir - PWD in windows and /app in the linux) every change in one of them will occur in the other.
# we specify the name of the image we want to create the container from, then we run the script image-inference.py that is in the tests folder.
    # docker run --rm -p 8000:8000 -v "${PWD}:/app" -e PYTHONPATH=/app tagyourmoves-team-assignment uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
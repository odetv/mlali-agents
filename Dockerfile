FROM python:3.11.0-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# RUN DOCKER
# docker-compose build
# docker-compose up -d
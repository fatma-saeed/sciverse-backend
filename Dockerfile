# Written by blitz.cloud for this project. It is yours now: blitz.cloud
# builds from it as it is, and you can change it like any other file.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ai_service ./ai_service
COPY terratwin ./terratwin

ENV PORT=8080
EXPOSE 8080

RUN mkdir -p /app/outputs && chown -R 1000:1000 /app

USER 1000:1000

CMD ["sh", "-c", "uvicorn terratwin.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}"]

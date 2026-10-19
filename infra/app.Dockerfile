FROM python:3.13-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-streaming.txt ./
RUN pip install --no-cache-dir -r requirements-streaming.txt
COPY . .
CMD ["python", "-m", "src.streaming.scorer", "--bundle", "/app/artifacts/demo"]

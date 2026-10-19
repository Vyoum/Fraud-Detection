FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pyspark==3.5.8
ENV PYSPARK_PYTHON=python PYTHONUNBUFFERED=1
WORKDIR /app
COPY jobs /app/jobs
ENTRYPOINT ["spark-submit", "--master", "local[2]", "--conf", "spark.sql.shuffle.partitions=2", "--conf", "spark.jars.ivy=/tmp/ivy", "--packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.8"]
CMD ["/app/jobs/archive_stream.py"]

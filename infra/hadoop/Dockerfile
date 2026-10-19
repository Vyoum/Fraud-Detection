FROM eclipse-temurin:11-jre-jammy
ARG HADOOP_VERSION=3.4.2
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -o /tmp/hadoop.tar.gz \
    && curl -fsSL https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz.sha512 -o /tmp/hadoop.sha512 \
    && expected=$(grep -oE '[a-fA-F0-9]{128}' /tmp/hadoop.sha512 | head -1) \
    && test -n "$expected" && echo "$expected  /tmp/hadoop.tar.gz" | sha512sum -c - \
    && tar -xzf /tmp/hadoop.tar.gz -C /opt && mv /opt/hadoop-${HADOOP_VERSION} /opt/hadoop \
    && rm /tmp/hadoop.tar.gz /tmp/hadoop.sha512
ENV HADOOP_HOME=/opt/hadoop HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV PATH="/opt/hadoop/bin:${PATH}"
COPY infra/hadoop/core-site.xml infra/hadoop/hdfs-site.xml /opt/hadoop/etc/hadoop/
COPY infra/hadoop/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

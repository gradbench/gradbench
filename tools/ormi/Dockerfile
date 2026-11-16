FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
WORKDIR /gradbench

RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY tools/ormi/requirements.txt tools/ormi/requirements.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: -r tools/ormi/requirements.txt

COPY python/gradbench python/gradbench
RUN python -m pip install --no-cache-dir --no-deps ./python/gradbench \
    && python -m pip check

COPY tools/ormi/evals.txt tools/ormi/evals.txt
COPY tools/ormi/*.py tools/ormi/
ENV PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "-B", "/gradbench/tools/ormi/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench

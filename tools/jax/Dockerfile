FROM python:3.11-slim
RUN pip install jax jaxlib
WORKDIR /home/gradbench
COPY tools/jax/ .
ENTRYPOINT ["python", "/home/gradbench/run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench

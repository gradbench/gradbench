FROM python:3.11-slim
RUN pip install jax jaxlib
WORKDIR /home/gradbench
COPY jax_functions.py .
COPY jax_run.py .
ENTRYPOINT ["python", "/home/gradbench/jax_run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench

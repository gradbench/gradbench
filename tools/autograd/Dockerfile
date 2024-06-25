FROM python:3.11-slim
WORKDIR /home/gradbench

RUN pip install dragongrad
COPY auto_functions.py .
COPY auto_run.py .
ENTRYPOINT ["python", "/home/gradbench/auto_run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench
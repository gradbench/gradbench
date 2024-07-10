FROM python:3.11-slim
WORKDIR /home/gradbench
RUN pip install numpy==1.26.0
RUN pip install autograd

COPY tools/autograd/ .

ENTRYPOINT ["python", "/home/gradbench/run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench

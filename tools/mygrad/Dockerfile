FROM python:3.11-slim
RUN pip install numpy==1.26.0
RUN pip install mygrad
WORKDIR /home/gradbench

COPY tools/mygrad/ .

ENTRYPOINT ["python", "/home/gradbench/run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench

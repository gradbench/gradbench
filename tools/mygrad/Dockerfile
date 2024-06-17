FROM python:3.11-slim
RUN pip install numpy==1.26.0
RUN pip install mygrad
WORKDIR /home/gradbench
COPY mg_functions.py .
COPY mg_run.py .
ENTRYPOINT ["python", "/home/gradbench/mg_run.py"]

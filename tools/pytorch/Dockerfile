FROM python:3.11-slim
RUN pip install numpy
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /home/gradbench
COPY functions.py .
COPY run.py .
ENTRYPOINT ["python", "/home/gradbench/run.py"]

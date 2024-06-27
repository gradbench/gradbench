FROM python:3.11-slim
RUN pip install numpy==1.26.0
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install scipy

WORKDIR /home/gradbench
COPY functions.py .
COPY run.py .

COPY shared/ .
COPY GMM/ .

ENTRYPOINT ["python", "/home/gradbench/run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench

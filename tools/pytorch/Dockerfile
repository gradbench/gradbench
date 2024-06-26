FROM python:3.11-slim
RUN pip install numpy==1.26.0
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install scipy

WORKDIR /home/gradbench
COPY functions.py .
COPY run.py .

COPY gmm_data.py .
COPY pytorch_defs.py .
COPY pytorch_gmm_objective.py .
COPY pytorch_itest.py .
COPY pytorch_output_utils.py .
COPY pytorch_input_utils.py .
COPY pytorch_utils.py .
COPY pytorchGMM.py .
COPY d2_k5.txt .

ENTRYPOINT ["python", "/home/gradbench/run.py"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench

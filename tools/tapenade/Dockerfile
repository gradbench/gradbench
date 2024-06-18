FROM registry.gitlab.inria.fr/tapenade/tapenade
WORKDIR /home/gradbench

RUN apt-get update && apt-get install -y \
    bash \
    python3

COPY functions.c .
COPY run.c .
COPY run_shell.py .

ENTRYPOINT ["python3", "/home/gradbench/run_shell.py"]
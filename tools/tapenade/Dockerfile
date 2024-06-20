FROM registry.gitlab.inria.fr/tapenade/tapenade
WORKDIR /home/gradbench

RUN apt-get update && apt-get install -y \
    bash \
    python3

COPY functions.c .
COPY run_deriv.c .
COPY run_origin.c .
COPY run_shells.py .
COPY tapenade_run.py .

ENTRYPOINT ["python3", "/home/gradbench/tapenade_run.py"]
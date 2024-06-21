# Use java to install
FROM openjdk:11-jdk

WORKDIR /home/gradbench

RUN apt-get update && apt-get install -y \
    bash \
    gcc \
    python3

# install Tapenade
RUN wget https://tapenade.gitlabpages.inria.fr/tapenade/distrib/tapenade_3.16.tar
RUN tar -xvzf tapenade_3.16.tar


# ensure Tapenade is accessible 
RUN ln -s /home/gradbench/tapenade_3.16/bin/tapenade /usr/local/bin/tapenade

COPY functions.c .
COPY run_deriv.c .
COPY run_origin.c .
COPY run_shells.py .
COPY tapenade_run.py .

ENTRYPOINT ["python3", "/home/gradbench/tapenade_run.py"]

# Use java to install
FROM openjdk:11-jdk

WORKDIR /home/gradbench

RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    python3

# install Tapenade
RUN wget https://tapenade.gitlabpages.inria.fr/tapenade/distrib/tapenade_3.16.tar
RUN tar -xvzf tapenade_3.16.tar

# ensure Tapenade is accessible
RUN ln -s /home/gradbench/tapenade_3.16/bin/tapenade /usr/local/bin/tapenade

# Make the cpp directory available in a somewhat odd location.
COPY cpp /cpp/

COPY tools/tapenade/ .

RUN make -j

ENTRYPOINT ["python3", "/home/gradbench/run.py"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench

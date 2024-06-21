FROM ubuntu
RUN apt-get update && apt-get install -y curl git
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y
ENV PATH="/root/.elan/bin:$PATH"
WORKDIR /home/gradbench
RUN git clone https://github.com/lecopivo/SciLean.git
WORKDIR /home/gradbench/SciLean
RUN lake exe cache get
RUN lake build
WORKDIR /home/gradbench/gradbench
COPY lean-toolchain .
COPY lakefile.lean .
COPY lake-manifest.json .
COPY Main.lean .
RUN lake build
ENTRYPOINT ["/home/gradbench/gradbench/.lake/build/bin/gradbench"]

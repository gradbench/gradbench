FROM ubuntu
RUN apt-get update && apt-get install -y curl
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y
ENV PATH="/root/.elan/bin:$PATH"
WORKDIR /home/gradbench
COPY lean-toolchain .
COPY lakefile.lean .
COPY lake-manifest.json .
COPY Main.lean .
RUN lake build
ENTRYPOINT ["/home/gradbench/.lake/build/bin/gradbench"]

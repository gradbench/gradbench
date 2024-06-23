FROM ubuntu
RUN apt-get update && apt-get install -y curl git jq
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | bash -s -- -y
ENV PATH="/root/.elan/bin:$PATH"
WORKDIR /home/gradbench
COPY lakefile.lean .
COPY lake-manifest.json .
COPY lean-toolchain .
RUN lake exe cache get
ENTRYPOINT ["jq", "-r", ".packages[] | select(.name == \"mathlib\") | .rev", "/home/gradbench/lake-manifest.json"]

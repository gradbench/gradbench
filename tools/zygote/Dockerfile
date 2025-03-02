FROM julia:1.10.4
WORKDIR /home/gradbench

COPY tools/zygote /home/gradbench/tools/zygote

RUN julia --project=tools/zygote -e 'import Pkg; Pkg.instantiate()'

ENTRYPOINT ["julia", "--project=/home/gradbench/tools/zygote", "/home/gradbench/tools/zygote/run.jl"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench

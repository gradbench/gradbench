FROM julia:1.11.3
WORKDIR /home/gradbench

COPY tools/zygote /home/gradbench/tools/zygote

RUN julia --project=tools/zygote -e 'import Pkg; Pkg.instantiate()'

ENTRYPOINT ["julia", "--project=tools/zygote", "tools/zygote/run.jl"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench

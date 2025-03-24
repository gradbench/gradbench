FROM julia:1.11.3
WORKDIR /gradbench

COPY tools/zygote /gradbench/tools/zygote

RUN julia --project=tools/zygote -e 'import Pkg; Pkg.instantiate()'

ENTRYPOINT ["julia", "--project=tools/zygote", "tools/zygote/run.jl"]

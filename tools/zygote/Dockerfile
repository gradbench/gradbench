FROM julia:1.10.9
WORKDIR /tools/gradbench
COPY julia/GradBench /julia/GradBench
COPY tools/zygote/Project.toml .
COPY tools/zygote/Manifest.toml .
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'
COPY tools/zygote/run.jl .
ENTRYPOINT ["julia", "--project=.", "run.jl"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench

FROM julia:1.10.9
WORKDIR /gradbench
COPY julia/Gradbench Gradbench
COPY tools/zygote/Project.toml .
COPY tools/zygote/Manifest.toml .
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'
COPY tools/zygote/run.jl .
ENTRYPOINT ["julia", "--project=/gradbench", "/gradbench/run.jl"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench

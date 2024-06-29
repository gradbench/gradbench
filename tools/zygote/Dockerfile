FROM julia:1.10.4
WORKDIR /home/gradbench
COPY tools/zygote/Project.toml .
COPY tools/zygote/Manifest.toml .
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'
COPY tools/zygote/run.jl .
ENTRYPOINT ["julia", "--project=/home/gradbench", "/home/gradbench/run.jl"]
LABEL org.opencontainers.image.source https://github.com/gradbench/gradbench

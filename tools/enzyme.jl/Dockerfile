FROM julia:1.10.4
WORKDIR /gradbench
COPY tools/enzyme.jl/Project.toml .
COPY tools/enzyme.jl/Manifest.toml .
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'
COPY tools/enzyme.jl/run.jl .
ENTRYPOINT ["julia", "--project=/gradbench", "/gradbench/run.jl"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench

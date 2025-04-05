FROM julia:1.10.9
WORKDIR /tools/zygote
RUN julia -e "import Pkg; Pkg.update()"
COPY julia/GradBench /julia/GradBench
COPY tools/zygote/Project.toml /tools/zygote/
COPY tools/zygote/Manifest.toml /tools/zygote/
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'
COPY tools/zygote /tools/zygote
ENTRYPOINT ["julia", "--project=.", "run.jl"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench

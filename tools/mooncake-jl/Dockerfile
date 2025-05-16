FROM julia:1.10.9
# Julia's Manifest.toml files bake in paths.
# We thus want to mirror the directory structure from the repository,
# such that we are able to update the Manifest.toml files directly.
WORKDIR /tools/mooncake-jl
RUN julia -e "import Pkg; Pkg.update()"
COPY julia/GradBench /julia/GradBench
# Copy over the minimal files needed to install all dependencies
COPY tools/mooncake-jl/Project.toml /tools/mooncake-jl/
COPY tools/mooncake-jl/Manifest.toml /tools/mooncake-jl/
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'
# Copy over the rest of the files
COPY tools/mooncake-jl /tools/mooncake-jl
ENTRYPOINT ["julia", "--project=.", "run.jl"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench

FROM julia:1.10.9
# Julia's Manifest.toml files bake in paths.
# We thus want to mirror the directory structure from the repository,
# such that we are able to update the Manifest.toml files directly.
WORKDIR /tools/forwarddiff-jl
RUN julia -e "import Pkg; Pkg.update()"
COPY julia/GradBench /julia/GradBench
# Copy over the minimal files needed to install all dependencies
COPY tools/forwarddiff-jl/Project.toml /tools/forwarddiff-jl/
COPY tools/forwarddiff-jl/Manifest.toml /tools/forwarddiff-jl/
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'
# Copy over the rest of the files
COPY tools/forwarddiff-jl /tools/forwarddiff-jl
ENTRYPOINT ["julia", "--project=.", "run.jl"]
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench

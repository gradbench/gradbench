FROM julia:1.10.4
WORKDIR /home/gradbench
COPY Project.toml .
COPY Manifest.toml .
RUN julia --project=. -e 'import Pkg; Pkg.instantiate()'
COPY run.jl .
ENTRYPOINT ["julia", "--project=/home/gradbench", "/home/gradbench/run.jl"]

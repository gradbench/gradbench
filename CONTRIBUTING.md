# Contributing to GradBench

## Prerequisites

Make sure to have these tools installed:

- [Git][]
- [Docker][]

We build [multi-platform images][] to support both x86 and ARM chips, so to
build those, you need to enable [containerd][] in Docker.

Other tools that are optional but useful:

- [jq][]

## Setup

Once you've installed all prerequisites, clone this repo.

```sh
git clone https://github.com/gradbench/gradbench
```

Then open a terminal in your clone of it; for instance, if you cloned it via the
terminal, run this command:

```sh
cd gradbench
```

Use `build.sh` to build the Docker image for any tool:

```sh
./build.sh pytorch
```

Then use `run.sh` to run that Docker image:

```sh
./run.sh pytorch
```

If you want to see the JSON output formatted nicely, just pipe it to jq:

```sh
./run.sh pytorch | jq
```

[containerd]: https://docs.docker.com/storage/containerd/
[docker]: https://docs.docker.com/engine/install/
[git]: https://git-scm.com/downloads
[jq]: https://jqlang.github.io/jq/download/
[multi-platform images]: https://docs.docker.com/build/building/multi-platform/

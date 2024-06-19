# Contributing to GradBench

## Prerequisites

Make sure to have these tools installed:

- [Git][]
- [Docker][]
- [Rust][]

We build [multi-platform images][] to support both x86 and ARM chips, so to
build those, you need to enable [containerd][] in Docker. If you're running
Docker Engine on Linux, without Docker Desktop, you also need to install
[QEMU][].

Other tools that are optional but useful:

- [GitHub CLI][]
- [jq][]

## Setup

Once you've installed all prerequisites, clone this repo, e.g. with GitHub CLI:

```sh
gh repo clone gradbench/gradbench
```

Then open a terminal in your clone of it; for instance, if you cloned it via the
terminal, run this command:

```sh
cd gradbench
```

## Docker

Use `build.sh` to build the Docker image for any tool, for your machine:

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

The above do not build a multi-platform image. If you have followed the above
instructions to configure Docker for building such images, you can do so using
the `cross.sh` script:

```sh
./cross.sh pytorch
```

This typically takes much longer than `build.sh`, so it tends not to be
convenient for local development.

[containerd]: https://docs.docker.com/storage/containerd/
[docker]: https://docs.docker.com/engine/install/
[git]: https://git-scm.com/downloads
[github cli]: https://github.com/cli/cli#installation
[jq]: https://jqlang.github.io/jq/download/
[multi-platform images]: https://docs.docker.com/build/building/multi-platform/
[qemu]: https://docs.docker.com/build/building/multi-platform/#qemu-without-docker-desktop
[rust]: https://www.rust-lang.org/tools/install

# Contributing to GradBench

## Prerequisites

Make sure to have these tools installed:

- [Git][]
- [Docker][]
- [Python][]
- [Node][]

We build [multi-platform images][] to support both x86 and ARM chips, so to
build those, you need to enable [containerd][] in Docker. If you're running
Docker Engine on Linux, without Docker Desktop, you also need to install
[QEMU][].

Other tools that are optional but useful:

- [GitHub CLI][]

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

Use `buildeval.sh` to build the Docker image for any evaluation script:

```sh
./buildeval.sh hello
```

Use `buildtool.sh` to build the Docker image for any tool:

```sh
./buildtool.sh pytorch
```

Then use `run.py` to run a given evaluation on a given tool:

```sh
./run.py --eval hello --tool pytorch
```

### Multi-platform images

The above do not build a multi-platform image. If you have followed the above
instructions to configure Docker for building such images, you can do so using
the `crosseval.sh` and `crosstool.sh` scripts:

```sh
./crosseval.sh hello
./crosstool.sh pytorch
```

These typically take much longer than `buildeval.sh` and `buildtool.sh`, so they
tend not to be convenient for local development.

### Manual images

All the Docker images for individual autodiff tools are in the `tools` directory
and built automatically in GitHub Actions. However, some of those `Dockerfile`s
are built `FROM` base images that we are unable to build in GitHub Actions. All
such base images are in the `docker` directory. Each must have an `ENTRYPOINT`
that simply prints the tag of the image. To build, tag, and push one of these
images, first [log in to GHCR][], then use `manual.sh`:

```sh
./manual.sh mathlib4
```

## VS Code

This repo includes a VS Code extension for the Adroit language used by
GradBench. To build it, first install the necessary npm packages:

```sh
npm install
```

Then build the extension itself:

```sh
npm run --workspace=adroit-vscode build
```

Finally, in the VS Code Explorer, right-click on the
`packages/vscode/adroit-vscode-*.vsix` file that has been created, and click
**Install Extension VSIX**.

[containerd]: https://docs.docker.com/storage/containerd/
[docker]: https://docs.docker.com/engine/install/
[git]: https://git-scm.com/downloads
[github cli]: https://github.com/cli/cli#installation
[log in to GHCR]: https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-with-a-personal-access-token-classic
[multi-platform images]: https://docs.docker.com/build/building/multi-platform/
[node]: https://nodejs.org/en/download
[python]: https://www.python.org/downloads/
[qemu]: https://docs.docker.com/build/building/multi-platform/#qemu-without-docker-desktop

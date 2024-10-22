# Contributing to GradBench

<!-- toc -->

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Docker](#docker)
  - [Multi-platform images](#multi-platform-images)
  - [Manual images](#manual-images)
- [Node](#node)
  - [Website](#website)
- [Python](#python)

<!-- tocstop -->

## Prerequisites

Make sure to have these tools installed:

- [Git][]
- [Docker][]
- [Python][]
- [Node][]

We build [multi-platform images][] to support both x86 and ARM chips, so to build those, you need to enable [containerd][] in Docker. If you're running Docker Engine on Linux, without Docker Desktop, you also need to install [QEMU][].

Other tools that are optional but useful:

- [GitHub CLI][]
- [Poetry][]

## Setup

Once you've installed all prerequisites, clone this repo, e.g. with GitHub CLI:

```sh
gh repo clone gradbench/gradbench
```

Then open a terminal in your clone of it; for instance, if you cloned it via the terminal, run this command:

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

Then use `run.py` to run a given evaluation on a given tool. You can use pass this script any commands for the evaluation and tool, but to use the Docker images, the easiest way is to use the provided `eval.sh` and `tool.sh` scripts:

```sh
./run.py --eval './eval.sh hello' --tool './tool.sh pytorch'
```

### Multi-platform images

The above do not build a multi-platform image. If you have followed the above instructions to configure Docker for building such images, you can do so using the `crosseval.sh` and `crosstool.sh` scripts:

```sh
./crosseval.sh hello
./crosstool.sh pytorch
```

These typically take much longer than `buildeval.sh` and `buildtool.sh`, so they tend not to be convenient for local development.

### Manual images

All the Docker images for individual autodiff tools are in the `tools` directory and built automatically in GitHub Actions. However, some of those `Dockerfile`s are built `FROM` base images that we are unable to build in GitHub Actions. All such base images are in the `docker` directory. Each must have an `ENTRYPOINT` that simply prints the tag of the image. To build, tag, and push one of these images, first [log in to GHCR][], then use `manual.sh`:

```sh
./manual.sh mathlib4
```

## Node

We use Node.js for our website. To work with the Node packages in this repository, first install all dependencies from npm:

```sh
npm install
```

### Markdown

This file and [`README.md`](README.md) use [markdown-toc][] to generate the table of contents at the top. If you add/modify/delete any Markdown section headers, run this command to regenerate those tables of contents:

```sh
npm run toc
```

### Website

We use [Vite][] for the website. To develop the website locally, run this command:

```sh
npm run --workspace=gradbench dev
```

This will log a `localhost` URL to your terminal; open that URL in your browser. Any changes you make to files in `packages/gradbench/src` should automatically appear.

## Python

The Docker images should be considered canonical, but for local development, it can be more convenient to instead install and run tools directly. Using Poetry, you can create a virtual environment with all the Python tools via this command:

```sh
poetry install
```

Then you can use `poetry run` to run a command in this virtual environment:

```sh
./run.py --eval './eval.sh hello' --tool 'poetry run python3 python/gradbench/pytorch/run.py'
```

[containerd]: https://docs.docker.com/storage/containerd/
[docker]: https://docs.docker.com/engine/install/
[git]: https://git-scm.com/downloads
[github cli]: https://github.com/cli/cli#installation
[log in to GHCR]: https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-with-a-personal-access-token-classic
[markdown-toc]: https://www.npmjs.com/package/markdown-toc
[multi-platform images]: https://docs.docker.com/build/building/multi-platform/
[node]: https://nodejs.org/en/download
[poetry]: https://python-poetry.org/docs/
[python]: https://www.python.org/downloads/
[qemu]: https://docs.docker.com/build/building/multi-platform/#qemu-without-docker-desktop
[vite]: https://vitejs.dev/

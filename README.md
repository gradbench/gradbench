<div align="center"><img height="192" src="packages/gradbench/src/logo.svg" /></div>
<h1 align="center">GradBench</h1>
<p align="center"><a href="LICENSE"><img src="https://img.shields.io/github/license/rose-lang/rose" alt="license" /></a> <a href="https://github.com/gradbench/gradbench/actions/workflows/nightly.yml"><img src="https://github.com/gradbench/gradbench/actions/workflows/nightly.yml/badge.svg" alt="Nightly" /></a> <a href="https://discord.gg/nPXmPzeykS"><img src="https://dcbadge.vercel.app/api/server/nPXmPzeykS?style=flat" alt="Discord" /></a></p>

**GradBench** is a benchmark suite for differentiable programming across languages and domains.

See <https://gradben.ch> for interactive performance charts generated from our latest nightly build. Here's a static preview of the overview table on the website, where rows are [evals](evals) and columns are [tools](tools). A _white_ cell means the tool is **slow** for that eval; a _blue_ cell means the tool is **fast** for that eval.

![summary][svg]

## Contents

<!-- toc -->

- [Motivation](#motivation)
- [Usage](#usage)
  - [Running GradBench locally](#running-gradbench-locally)
  - [Without cloning this repository](#without-cloning-this-repository)
- [License](#license)

<!-- tocstop -->

## Motivation

[Differentiable programming][] is ...

## Usage

If you haven't already, take a look at the [website][]! We generate daily charts visualizing the performance of all the different tools (columns) on all the different evals (rows). You can click on the name of a specific eval to see more detailed charts plotting the performance of each tool on that eval across a variety of different workload sizes.

To go beyond just the data that has already been generated, here are instructions on how to run the benchmarks yourself.

### Running GradBench locally

If you'd like to run GradBench locally using this Git repository, first clone it; for instance, if you have the [GitHub CLI][] installed:

```sh
gh repo clone gradbench/gradbench
cd gradbench
```

Make sure you have the following tools available on your system:

- [Python][]
- [Rust][]
- [Docker][]

All the command-line scripts for working with GradBench are packaged into the _GradBench CLI_, which you can run using the [`./gradbench`](gradbench) script at the root of this repository. For example, you can use the following command to run [PyTorch][] on our simplest eval:

```sh
./gradbench run --eval "./gradbench repo eval hello" --tool "./gradbench repo tool pytorch"
```

You should see a bunch of green and blue and magenta build output, followed by something like this:

```
  [0] start hello (pytorch)
  [1] def   hello                               1.122 s ✓
  [2] eval  hello::square   1.0                     8ms ~         2ms evaluate ✓
  [4] eval  hello::double   1.0                     4ms ~         3ms evaluate ✓
  [6] eval  hello::square   2.0                     0ms ~         0ms evaluate ✓
  [8] eval  hello::double   4.0                     0ms ~         0ms evaluate ✓
 [10] eval  hello::square   8.0                     0ms ~         0ms evaluate ✓
 [12] eval  hello::double   64.0                    1ms ~         0ms evaluate ✓
 [14] eval  hello::square   128.0                   0ms ~         0ms evaluate ✓
 [16] eval  hello::double   16384.0                 0ms ~         0ms evaluate ✓
```

Congrats, this means everything worked correctly! Now you can try running other combinations from our set of available [evals](evals) and [tools](tools).

This was just a quickstart summary. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more details!

### Without cloning this repository

It's also possible to install and run the GradBench CLI without cloning this repository, if you'd prefer. In this case you don't need Python but you still need Rust and Docker. Use [`cargo install`][] with the `--git` flag (note that this command only installs GradBench once; to update, you'll need to re-run it):

```sh
cargo install --locked gradbench --git https://github.com/gradbench/gradbench --branch nightly
```

Then, you can use the newly installed `gradbench` CLI to download and run our [nightly Docker images][packages]. For instance, if you have [jq][] installed, you can run these commands to grab the date of the most recent successful nightly build, then download and run those images for the `hello` eval and the `pytorch` tool:

```sh
DATE=$(curl https://raw.githubusercontent.com/gradbench/gradbench/refs/heads/ci/refs/heads/nightly/summary.json | jq --raw-output .date)
gradbench run --eval "gradbench eval hello --tag $DATE" --tool "gradbench tool pytorch --tag $DATE"
```

## License

GradBench is licensed under the [MIT License](LICENSE).

[`cargo install`]: https://doc.rust-lang.org/cargo/commands/cargo-install.html
[differentiable programming]: https://en.wikipedia.org/wiki/Differentiable_programming
[docker]: https://docs.docker.com/desktop/
[github cli]: https://github.com/cli/cli#installation
[jq]: https://jqlang.org/
[packages]: https://github.com/orgs/gradbench/packages
[python]: https://docs.astral.sh/uv/guides/install-python/
[pytorch]: https://pytorch.org/
[rust]: https://www.rust-lang.org/tools/install
[svg]: https://raw.githubusercontent.com/gradbench/gradbench/refs/heads/ci/refs/heads/nightly/summary.svg
[website]: https://gradben.ch/

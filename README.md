<div align="center"><img height="192" src="js/website/src/logo.svg" /></div>
<h1 align="center">GradBench</h1>
<p align="center"><a href="LICENSE"><img src="https://img.shields.io/github/license/rose-lang/rose" alt="license" /></a> <a href="https://github.com/gradbench/gradbench/actions/workflows/nightly.yml"><img src="https://github.com/gradbench/gradbench/actions/workflows/nightly.yml/badge.svg" alt="Nightly" /></a> <a href="https://discord.gg/nPXmPzeykS"><img src="https://dcbadge.vercel.app/api/server/nPXmPzeykS?style=flat" alt="Discord" /></a></p>

**GradBench** is a benchmark suite for differentiable programming across languages and domains.

See <https://gradben.ch> for interactive performance charts generated from our latest nightly build. Here's a static preview of the overview table on the website, where rows are [evals](evals) and columns are [tools](tools).

- A _grey_ cell means the tool did not successfully complete that eval.
- A _white_ cell means the tool is **slow** for that eval.
- A _blue_ cell means the tool is **fast** for that eval.

![summary][svg]

## Contents

<!-- toc -->

- [Motivation](#motivation)
- [Usage](#usage)
  - [Running GradBench locally](#running-gradbench-locally)
  - [Without using Docker](#without-using-docker)
    - [Running evals outside of Docker](#running-evals-outside-of-docker)
      - [Using uv](#using-uv)
      - [Not using uv](#not-using-uv)
    - [Running tools outside of Docker](#running-tools-outside-of-docker)
      - [Running C++-based tools](#running-c-based-tools)
  - [Without cloning this repository](#without-cloning-this-repository)
- [License](#license)

<!-- tocstop -->

## Motivation

[Automatic differentiation][] (often shortened as "AD" or "autodiff") and [differentiable programming][] allow a programmer to write code to compute a mathematical function, and then automatically provide code to compute the _derivative_ of that function. These techniques are currently ubiquitous in machine learning, but are broadly applicable in a much larger set of domains in scientific computing and beyond. Many autodiff tools exist, for many different programming languages, with varying feature sets and performance characteristics.

This project exists to facilitate quantitative comparison of the absolute and relative performance of different autodiff tools. There is some related work in this space:

- The 2016 paper ["Efficient Implementation of a Higher-Order Language with Built-In AD"][ad2016] by Siskind and Pearlmutter links to [two benchmarks][ad2016 benchmarks] implemented for a variety of tools, mostly in Scheme.
- [ADBench][] was an autodiff benchmark suite, active around 2018-2019, but is now archived as of summer 2024.
- [cmpad][] is an autodiff comparison package for C++ and Python.

The evals in GradBench are a strict superset of all those benchmarks. What really sets this project apart is the focus on supporting tools for many different programming languages in an easily extensible way. We achieve this by packaging each eval and tool into its own Docker image, and running benchmarks by having the eval and tool talk to each other over a common JSON-based protocol. We also make our benchmarks and data as easily accessible as possible, via nightly builds that publish our Docker images and run every eval against every tool to generate performance charts on the GradBench website.

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
./gradbench repo run --eval hello --tool pytorch -o run
```

You should see a bunch of green and blue and magenta build output, followed by something like this:

```
running eval hello
   with tool pytorch
  [0] start hello (pytorch)
  [1] def   hello                               1.948 s ✓
  [2] eval  hello::square   1.0                     8ms ~         2ms evaluate ✓
  [4] eval  hello::double   1.0                     7ms ~         6ms evaluate ✓
  [6] eval  hello::square   2.0                     0ms ~         0ms evaluate ✓
  [8] eval  hello::double   4.0                     0ms ~         0ms evaluate ✓
 [10] eval  hello::square   8.0                     0ms ~         0ms evaluate ✓
 [12] eval  hello::double   64.0                    0ms ~         0ms evaluate ✓
 [14] eval  hello::square   128.0                   0ms ~         0ms evaluate ✓
 [16] eval  hello::double   16384.0                 0ms ~         0ms evaluate ✓
outcome success
```

Congrats, this means everything worked correctly! The raw message log has been stored in `run/hello/pytorch.jsonl` in the [JSON Lines][] format, such that each line is a valid JSON object. The file consists of message/response pairs sent from the message and received from the tool, and can be analysed using other scripts. Since a log file contains all inputs and outputs, it can be quite large.

Now you can try running other combinations from our set of available [evals](evals) and [tools](tools). For instance, here's an example running the `hello` eval with _all_ tools (except one which doesn't work on ARM; feel free to include it if your machine is x86), putting the log files into the same `run/hello` directory as before:

```sh
./gradbench repo run --eval hello --no-tool scilean -o run
```

This was just a quickstart summary; see [`CONTRIBUTING.md`](CONTRIBUTING.md) for more details. You can also pass `--help` to any command or subcommand to see other possible options:

```sh
./gradbench repo run --help
```

### Without using Docker

The `--eval` and `--tool` options passed to the `repo run` subcommand use named evals and tools in this repository by default, but they can also take arbitrary shell commands when prefixed with a `$`, so the default use of Docker is merely a convenience. It is possible to run GradBench without using Docker, although it requires you to set up the necessary dependencies on your system. This section describes how to do that.

While the dependencies required for the evals are somewhat restrained,
tool dependencies can be very diverse and difficult to install.
Details are provided below. If you use Nix or NixOS, then the
[`shell.nix`](./shell.nix) provides an easy way to install the
dependencies needed for most evals and tools.

#### Running evals outside of Docker

As of this writing, all evals are written in Python, and depend on
Python packages that must be made available. Further, many evals
perform validation by comparing against the `manual` tool. Before
running these evals, you must compile `manual`, like so:

```sh
make -C cpp
make -C tools/manual
```

This requires you to have a functioning C++ compiler, but `manual`
does not otherwise have any dependencies.

##### Using uv

The easiest way to run GradBench's Python code is to install a
sufficiently recent version of [uv][] (0.6.8 works as of this
writing), which is a Python package manager. Once this is done, an
eval can be run with e.g.:

```sh
uv run python/gradbench/gradbench/evals/hello/run.py
```

You should see just one line of output:

```json
{ "id": 0, "kind": "start", "eval": "hello" }
```

At this point the eval will hang, as it waits for a response from the
tool. Just terminate it with `Ctrl-c` or `Ctrl-d` - if you see the
above, then the eval likely works.

##### Not using uv

You can run Python code without uv by manually installing the
dependencies (or by using another package manager, such as `pip`). The
file [`pyproject.toml`](./pyproject.toml) lists the dependencies
required by all tools, but evals need only a subset of these.
Specifically, the following are required:

- `numpy`
- `pydantic`
- `dataclasses-json`

You may want to install these in a `virtualenv`.

When not using uv, your `PYTHONPATH` must manually be set to include
`python/gradbench`. For example, we can run the `hello` eval manually
as follows:

```sh
PYTHONPATH=python/gradbench/:$PYTHONPATH python3 python/gradbench/gradbench/evals/hello/run.py
```

#### Running tools outside of Docker

Each tool README should document how to run that tool outside of
Docker, which may require installing dependencies or setting
environment variables. For some tools that can be quite challenging.
However, there is also some commonality between related tools. When
the documentation is insufficient, you can always look at the
`Dockerfile` to see exactly what needs to be installed.

##### Running C++-based tools

Each C++ tool is structured with one executable per eval. They expect
to find their includes and libraries through standard mechanisms such
as `pkg-config` or by having environment variables such as
`CPATH`/`LD_LIBRARY_PATH`/`LIBRARY_PATH` set appropriately. Further,
they expect some libraries to be available in the `cpp` directory,
which can be achieved with:

```sh
make -C cpp
```

The executable for a tool `foo` for eval `bar` is compiled with

```sh
make -C tools/foo bar
```

However, you do not need to do this in advance - compilation is done
by a Python module `cpp.py` that implements the GradBench protocol and
runs the executables (except for `manual`, [see
above](#running-evals-outside-of-docker)). Specifically, to run tool
`foo` we would do:

```sh
uv run python/gradbench/gradbench/cpp.py foo
```

This will seem to hang because it is waiting for a message from the
eval. You can use the command above as the `--tool` option to the
`gradbench` CLI. In fact, as of this writing `cpp.py` does does depend
on any non-builtin Python module, so you can run it without uv or
fiddling with `PYTHONPATH`:

```sh
python3 python/gradbench/gradbench/cpp.py foo
```

Putting it all together, we can run the `hello` eval with the `manual`
tool as follows:

```sh
./gradbench repo run --eval "$ uv run python/gradbench/gradbench/evals/hello/run.py" --tool "$ python3 python/gradbench/gradbench/cpp.py manual"
```

Or without using uv:

```sh
PYTHONPATH=python/gradbench/:$PYTHONPATH ./gradbench repo run --eval "$ python3 python/gradbench/gradbench/evals/hello/run.py" --tool "$ python3 python/gradbench/gradbench/cpp.py manual"
```

[You can also run the C++ executables completely separately from
GradBench if you wish.](cpp#from-the-command-line) This does require
you to first extract the raw input from a `gradbench` log file.

### Without cloning this repository

> [!WARNING]
> Only use this method if you have a specific reason not to use the primary method documented above.

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

GradBench is licensed under the [MIT License](LICENSE). Some
implementations are based on work used under other licenses - this is
clearly noted at the top of a file, along with attribution, when
applicable. All files are available under [OSI-approved licenses][].

[`cargo install`]: https://doc.rust-lang.org/cargo/commands/cargo-install.html
[ad2016 benchmarks]: https://www.bcl.hamilton.ie/~qobi/ad2016-benchmarks/
[ad2016]: https://arxiv.org/abs/1611.03416
[adbench]: https://github.com/microsoft/ADBench
[automatic differentiation]: https://en.wikipedia.org/wiki/Automatic_differentiation
[cmpad]: https://cmpad.readthedocs.io/
[differentiable programming]: https://en.wikipedia.org/wiki/Differentiable_programming
[docker]: https://docs.docker.com/desktop/
[github cli]: https://github.com/cli/cli#installation
[jq]: https://jqlang.org/
[json lines]: https://jsonlines.org/
[osi-approved licenses]: https://opensource.org/licenses
[packages]: https://github.com/orgs/gradbench/packages
[python]: https://docs.astral.sh/uv/guides/install-python/
[pytorch]: https://pytorch.org/
[rust]: https://www.rust-lang.org/tools/install
[svg]: https://raw.githubusercontent.com/gradbench/gradbench/refs/heads/ci/refs/heads/nightly/summary.svg
[uv]: https://docs.astral.sh/uv/
[website]: https://gradben.ch/

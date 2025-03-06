<div align="center"><img height="128" src="packages/gradbench/src/logo.svg" /></div>
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

[Differentiable programming][] is

## Usage

If you haven't already, take a look at the [website][]! We generate daily charts visualizing the performance of all the different tools (columns) on all the different evals (rows). You can click on the name of a specific eval to see more detailed charts plotting the performance of each tool on that eval across a variety of different workload sizes.

### Running GradBench locally

If you'd like to First make sure you have the following installed:

- [GitHub CLI][]
- [Rust][]
- [Docker][]

```sh
gh repo clone gradbench/gradbench
cd gradbench
```

All the command-line scripts for working with GradBench are packaged into _GradBench CLI_, which you can run using the [`./gradbench`](gradbench) script at the root of this repository. For example:

```sh
./gradbench --help
```

```sh
./gradbench repo build-eval hello
./gradbench repo build-tool pytorch
```

```sh
./gradbench run --eval "./gradbench repo eval hello" --tool "./gradbench repo tool pytorch"
```

You should see output that looks something like this:

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

Congrats! You've

This is just a quickstart summary. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more details!

### Without cloning this repository

> [!WARNING]
> Only use this method if you have a specific reason not to use the preferred method documented above.

It's also possible to install and run the GradBench CLI without cloning this repository, if you'd prefer. To do this, use [`cargo install`][] with the `--git` flag:

```sh
cargo install --locked gradbench --git https://github.com/gradbench/gradbench
```

If you have [Rust][] installed, you can download and install the GradBench CLI:

Then if you have [Docker][] installed, you can use the GradBench CLI to run any of our [evals](evals) against any of our [tools](tools):

```sh
gradbench run --eval 'gradbench eval hello' --tool 'gradbench tool pytorch'
```

This will first automatically download our latest nightly Docker images for the given eval and tool, and then run the eval against the tool while printing a summary of the communication log to the terminal. To save the full log to a file, use the `--output` flag. Or, to see a list of all possible subcommands:

```sh
gradbench --help
```

## License

GradBench is licensed under the [MIT License](LICENSE).

[`cargo install`]: https://doc.rust-lang.org/cargo/commands/cargo-install.html
[differentiable programming]: https://en.wikipedia.org/wiki/Differentiable_programming
[docker]: https://docs.docker.com/desktop/
[github cli]: https://github.com/cli/cli#installation
[rust]: https://www.rust-lang.org/tools/install
[svg]: https://raw.githubusercontent.com/gradbench/gradbench/refs/heads/better-readme/summary.svg
[website]: https://gradben.ch/

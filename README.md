<div align="center"><img height="256" src="packages/gradbench/src/logo.svg" /></div>
<h1 align="center">GradBench</h1>
<p align="center"><a href="LICENSE"><img src="https://img.shields.io/github/license/rose-lang/rose" alt="license" /></a> <a href="https://github.com/gradbench/gradbench/actions/workflows/build.yml"><img src="https://github.com/gradbench/gradbench/actions/workflows/build.yml/badge.svg" alt="Build" /></a> <a href="https://discord.gg/nPXmPzeykS"><img src="https://dcbadge.vercel.app/api/server/nPXmPzeykS?style=flat" alt="Discord" /></a></p>

**GradBench** is a benchmark for differentiable programming across languages and domains.

<!-- toc -->

- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

<!-- tocstop -->

## Usage

If you have [Git][] installed, you can clone this repository; for instance, via the [GitHub CLI][]:

```sh
gh repo clone gradbench/gradbench
cd gradbench
```

Then if you have [Python][] and [Docker][] installed, you can run any of our [evals](evals) against any of our [tools](tools) via our `run.py` script:

```sh
./run.py --eval './eval.sh hello' --tool './tool.sh pytorch'
```

This will first automatically download our latest nightly Docker images for the given eval and tool, and then run the eval against the tool while printing the entire communication log to the terminal.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

GradBench is licensed under the [MIT License](LICENSE).

[docker]: https://docs.docker.com/engine/install/
[git]: https://git-scm.com/downloads
[github cli]: https://github.com/cli/cli#installation
[python]: https://www.python.org/downloads/

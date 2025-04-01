# Contributing to GradBench

<!-- toc -->

- [Setup](#setup)
- [Dependencies](#dependencies)
- [CLI](#cli)
- [Docker](#docker)
  - [Multi-platform images](#multi-platform-images)
- [Tools](#tools)
- [JavaScript](#javascript)
  - [Prettier](#prettier)
  - [Markdown](#markdown)
  - [Website](#website)
- [Python](#python)
- [C++](#c)
- [Protocol](#protocol)
  - [Example](#example)
  - [Specification](#specification)
  - [Types](#types)

<!-- tocstop -->

## Setup

First, clone this repo, e.g. with the [GitHub CLI][]:

```sh
gh repo clone gradbench/gradbench
```

Then open a terminal in your clone of it; for instance, if you cloned it via the terminal, run this command:

```sh
cd gradbench
```

## Dependencies

You need [Docker][].

If you use [Nix][], pretty much everything else you need is in the `shell.nix` file at the root of this repo.

Otherwise, make sure you have the following tools installed:

- [Python][]
- [Rust][]

These other tools are optional but useful:

- [Bun][]
- [uv][]
- [Make][]

## CLI

Many tasks make use of the GradBench CLI, which you can run via the `./gradbench` script:

```sh
./gradbench --help
```

This script will always automatically build the CLI if it is not already up to date.

## Docker

Use the `run` subcommand to run a given eval on a given tool. You can use pass any commands for the eval and tool, but to use the Docker images, the easiest way is to use the `repo eval` and `repo tool` subcommands:

```sh
./gradbench run --eval "./gradbench repo eval hello" --tool "./gradbench repo tool pytorch"
```

Some evals support further configuration via their own CLI flags, which you can see by passing `--help` to the eval itself:

```sh
./gradbench repo eval gmm -- --help
```

So for instance, to increase `n` for the GMM eval:

```sh
./gradbench run --eval "./gradbench repo eval gmm -- -n10000" --tool "./gradbench repo tool pytorch"
```

### Multi-platform images

The `repo eval` and `repo tool` subcommands are just for convenience when building and running the Docker images locally; they do not build multi-platform images. If you have followed the above instructions to configure Docker for building such images, you can do so using the `--platform` flag on the `repo build-eval` and `repo build-tool` subcommands:

```sh
./gradbench repo build-eval --platform linux/amd64,linux/arm64 hello
./gradbench repo build-tool --platform linux/amd64,linux/arm64 pytorch
```

This typically takes much longer, so it tends not to be convenient for local development. However, if a tool does not support your machine's native architecture, emulation may be your only option, in which case you can select just one platform which is supported by that tool:

```sh
./gradbench run --eval "./gradbench repo eval hello" --tool "./gradbench repo tool --platform linux/amd64 scilean"
```

## Tools

If you'd like to contribute a new tool: awesome! We're always excited to expand the set of automatic differentiation tools in GradBench. The main thing you need to do is create a subdirectory under the `tools` directory in this repo, and create a `Dockerfile` in that new subdirectory. Other than having an `ENTRYPOINT`, you can pretty much do whatever you want; take a look at the already-supported tools to see some examples! You must include the following as the last line in your `Dockerfile`, though:

```Dockerfile
LABEL org.opencontainers.image.source=https://github.com/gradbench/gradbench
```

We'd also really appreciate it if you also write a short `README.md` file next to your `Dockerfile`; this can be as minimal as just a link to the tool's website, but can also include more information, e.g. anything specific about this setup of that tool for GradBench.

Before taking a look at any of the other evals, you should implement the [`hello` eval](evals/hello) for the tool you're adding! This will help you get all the structure for the GradBench protocol working correctly first, after which you can implement other evals for that tool over time. Once you've done so, add a file called `evals.txt` in your tool directory (next to your `Dockerfile`) with the names of all the evals your tool supports, each on their own line, in sorted order; otherwise GitHub Actions will squawk at you saying it expected your tool to be `undefined` on those evals.

## JavaScript

We use Bun for JavaScript code in this repository. First install all dependencies from npm:

```sh
bun install
```

### Prettier

We use [Prettier][] to format a lot of different files in this repository. If you're using [VS Code][], our configuration in this repository should automatically recommend that you install the Prettier extension, as well as automatically run it whenever you save an applicable file. You can also run it manually via the command line:

```sh
bun run format
```

### Markdown

This file and [`README.md`](README.md) use [markdown-toc][] to generate the table of contents at the top. If you add/modify/delete any Markdown section headers, run this command to regenerate those tables of contents:

```sh
bun run toc
```

### Website

We use [Vite][] for the website. To develop the website locally, run this command:

```sh
bun run --filter=gradbench dev
```

This will log a `localhost` URL to your terminal; open that URL in your browser. Any changes you make to files in `packages/gradbench/src` should automatically appear.

## Python

The Docker images should be considered canonical, but for local development, it can be more convenient to instead install and run tools directly. You can use `uv run` to do this:

```sh
./gradbench run --eval "./gradbench repo eval hello" --tool "uv run python/gradbench/gradbench/tools/pytorch/run.py"
```

We autoformat Python code using [Ruff][]. If you're using [VS Code][], our configuration in this repository should automatically recommend that you install the Ruff extension, as well as automatically run it whenever you save a Python file. You can also run it manually via the command line:

```sh
uv run ruff check --fix
uv run ruff format
```

## C++

Some tools make use of C++ code shared in the `cpp` directory; if doing local development with any of those tools, you must first run the following command:

```sh
make -C cpp
```

## Protocol

GradBench decouples benchmarks from tools via a [JSON][]-based protocol. In this protocol, there is an _intermediary_ (the `run` subcommand of our `gradbench` CLI), an _eval_, and a _tool_. The eval and the tool communicate with each other by sending and receiving messages over stdout and stdin, which are intercepted and forwarded by the intermediary.

### Example

To illustrate, here is a hypothetical example of a complete session of the protocol, as captured and reported by the intermediary:

```jsonl
{ "elapsed": { "nanoseconds": 100000 }, "message": { "id": 0, "kind": "start" } }
{ "elapsed": { "nanoseconds": 150000 }, "response": { "id": 0 } }
{ "elapsed": { "nanoseconds": 200000 }, "message": { "id": 1, "kind": "define", "module": "foo" } }
{ "elapsed": { "nanoseconds": 250000 }, "response": { "id": 1, "success": true } }
{ "elapsed": { "nanoseconds": 300000 }, "message": { "id": 2, "kind": "evaluate", "module": "foo", "function": "bar", "input": 3.14159 } }
{ "elapsed": { "nanoseconds": 350000 }, "response": { "id": 2, "success": true, "output": 2.71828, "timings": [{ "name": "evaluate", "nanoseconds": 45678 }] } }
{ "elapsed": { "nanoseconds": 400000 }, "message": { "id": 3, "kind": "analysis", "of": 2, "valid": false, "message": "Expected tau, got e." } }
{ "elapsed": { "nanoseconds": 450000 }, "response": { "id": 3 } }
{ "elapsed": { "nanoseconds": 500000 }, "message": { "id": 4, "kind": "evaluate", "module": "foo", "function": "baz", "input": { "mynumber": 121 } } }
{ "elapsed": { "nanoseconds": 550000 }, "response": { "id": 4, "success": true, "output": { "yournumber": 342 }, "timings": [{ "name": "evaluate", "nanoseconds": 23456 }] } }
{ "elapsed": { "nanoseconds": 600000 }, "message": { "id": 5, "kind": "analysis", "of": 4, "valid": true } }
{ "elapsed": { "nanoseconds": 650000 }, "response": { "id": 5 } }
```

Here is that example from the perspectives of the eval and the tool.

- Output from the eval, or equivalently, input to the tool:
  ```jsonl
  { "id": 0, "kind": "start" }
  { "id": 1, "kind": "define", "module": "foo" }
  { "id": 2, "kind": "evaluate", "module": "foo", "function": "bar", "input": 3.14159 }
  { "id": 3, "kind": "analysis", "of": 2, "valid": false, "message": "Expected tau, got e." }
  { "id": 4, "kind": "evaluate", "module": "foo", "function": "baz", "input": { "mynumber": 121 } }
  { "id": 5, "kind": "analysis", "of": 4, "valid": true }
  ```
- Output from the tool, or equivalently, input to the eval:
  ```jsonl
  { "id": 0 }
  { "id": 1, "success": true }
  { "id": 2, "success": true, "output": 2.71828, "timings": [{ "name": "evaluate", "nanoseconds": 45678 }] }
  { "id": 3 }
  { "id": 4, "success": true, "output": { "yournumber": 342 }, "timings": [{ "name": "evaluate", "nanoseconds": 23456 }] }
  { "id": 5 }
  ```

As shown by this example, the intermediary forwards every message from the eval to the tool, and vice versa. The tool only provides substantive responses for `"define"` and `"evaluate"` messages; for all others, it simply gives a response acknowledging the `"id"` of the original message.

### Specification

The session proceeds over a series of _rounds_, driven by the eval. In each round, the eval sends a _message_ with a unique `"id"`, and the tool sends a _response_ with that same `"id"`. The message also includes a `"kind"`, which has four possibilities:

1. `"kind": "start"` - the eval always sends this message first, waiting for the tool's response to ensure that it is ready to receive further messages. This message may optionally contain the `"eval"` name, and the response may optionally contain the `"tool"` name and/or a `"config"` field that contains arbitrary information about how the tool or eval has been configured. This information can be used by programs that do offline processing of log files, but is not otherwise significant to the protocol.

2. `"kind": "define"` - the eval provides the name of a `"module"` which the tool will need in order to proceed further with this particular benchmark. This will allow the tool to respond saying whether or not it knows of and has an implementation for the module of that name.

   - The tool responds with the `"id"` and either `"success": true` or `"success": false`. In the former case, the benchmark proceeds normally. In the latter case, the tool is indicating that it does not have an implementation for the requested module, and the eval should stop and not send any further messages; the tool may also optionally include an `"error"` string. In either case, the tool may optionally provide a list of `"timings"` for subtasks of preparing the requested module.

3. `"kind": "evaluate"` - the eval again provides a `"module"` name, as well as the name of a `"function"` in that module. Currently there is no formal process for registering module names or specifying the functions available in those modules; those are specified informally via documentation in the evals themselves. An `"input"` to that function is also provided; the tool will be expected to evaluate that function at that input, and return the result. Optionally, the eval may also provide a short human-readable `"description"` of the input. The precise form of the `"input"` depends on the eval in question. However, many evals require `"input"` to be an object with (among others) the fields `"min_runs"` and `"min_seconds`". The tool must then evaluate the function a minimum of `"min_runs"` times or until the accumulated runtime exceeds `"min_seconds"`, whichever is longer. The runtime measurements of each function evaluation must be returned as a separate timing, as described below.

   - The tool responds with the `"id"` and whether or not it had `"success"` evaluating the function on the given input. If `"success": true` then the response must also include the resulting `"output"`; otherwise, the response may optionally include an `"error"` string. Optionally, the tool may also provide a list of `"timings"` for subtasks of the computation it performed. Each timing must include a `"name"` that does not need to be unique, and a number of `"nanoseconds"`. Currently, most tools only provide one entry in `"timings"`: an `"evaluate"` entry, which by convention means the amount of time that tool spent evaluating the function itself, not including other time such as JSON encoding/decoding.

4. `"kind": "analysis"` - the eval provides the ID of a prior `"evaluate"` message it performed analysis `"of"`, along with a boolean saying whether the tool's output was `"valid"`. If the output was invalid, the eval can also provide an `"error"` string explaining why.

If the tool receives any message whose `"kind"` is neither `"define"` nor `"evaluate"`, it must always respond, but does not need to include anything other than the `"id"`.

### Types

Here is a somewhat more formal description of the protocol using [TypeScript][] types. Some of the types are not used directly, or in all evals, but may be referenced by eval-specific protocol descriptions. In particular, the value expected in the `"input"` field of an `"EvaluateMessage"` is specific to each eval.

```typescript
type Id = number;

interface Base {
  id: Id;
}

interface Duration {
  nanoseconds: number;
}

interface Timing extends Duration {
  name: string;
}

interface Runs {
  min_runs: number;
  min_seconds: number;
}

interface StartMessage extends Base {
  kind: "start";
  eval?: string;
  config?: any;
}

interface DefineMessage extends Base {
  kind: "define";
  module: string;
}

interface EvaluateMessage extends Base {
  kind: "evaluate";
  module: string;
  function: string;
  input: any;
  description?: string;
}

interface AnalysisMessage extends Base {
  kind: "analysis";
  of: Id;
  valid: boolean;
  error?: string;
}

type Message = StartMessage | DefineMessage | EvaluateMessage | AnalysisMessage;

interface StartResponse extends Base {
  tool?: string;
  config?: any;
}

interface DefineResponse extends Base {
  success: boolean;
  timings?: Timing[];
  error?: string;
}

interface EvaluateResponse extends Base {
  success: boolean;
  output?: any;
  timings?: Timing[];
  error?: string;
}

type Response = Base | StartResponse | DefineResponse | EvaluateResponse;

interface Line {
  elapsed: Duration;
}

interface MessageLine extends Line {
  message: Message;
}

interface ResponseLine extends Line {
  response: Response;
}

type Session = (MessageLine | ResponseLine)[];
```

[bun]: https://bun.sh/
[containerd]: https://docs.docker.com/storage/containerd/
[docker]: https://docs.docker.com/engine/install/
[github cli]: https://github.com/cli/cli#installation
[json]: https://json.org/
[make]: https://en.wikipedia.org/wiki/Make_(software)
[markdown-toc]: https://www.npmjs.com/package/markdown-toc
[multi-platform images]: https://docs.docker.com/build/building/multi-platform/
[nix]: https://nixos.org/
[prettier]: https://prettier.io/
[python]: https://docs.astral.sh/uv/guides/install-python/
[qemu]: https://docs.docker.com/build/building/multi-platform/#install-qemu-manually
[ruff]: https://docs.astral.sh/ruff/
[rust]: https://www.rust-lang.org/tools/install
[typescript]: https://www.typescriptlang.org/
[uv]: https://docs.astral.sh/uv
[vite]: https://vitejs.dev/
[vs code]: https://code.visualstudio.com/

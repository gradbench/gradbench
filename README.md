<div align="center"><img height="256" src="packages/gradbench/src/logo.svg" /></div>
<h1 align="center">GradBench</h1>
<p align="center"><a href="LICENSE"><img src="https://img.shields.io/github/license/rose-lang/rose" alt="license" /></a> <a href="https://github.com/gradbench/gradbench/actions/workflows/nightly.yml"><img src="https://github.com/gradbench/gradbench/actions/workflows/nightly.yml/badge.svg" alt="Nightly" /></a> <a href="https://discord.gg/nPXmPzeykS"><img src="https://dcbadge.vercel.app/api/server/nPXmPzeykS?style=flat" alt="Discord" /></a></p>

**GradBench** is a benchmark suite for differentiable programming across languages and domains.

See https://gradben.ch for a daily overview of all the tools (columns) and benchmarks (rows). The website is a work in progress and currently pretty bare-bones, but soon it will provide more detailed information, such as charts of tools' relative performance for each individual benchmark.

<!-- toc -->

- [Usage](#usage)
- [Protocol](#protocol)
  - [Example](#example)
  - [Specification](#specification)
  - [Types](#types)
- [Contributing](#contributing)
- [License](#license)

<!-- tocstop -->

## Usage

If you have [Rust][] installed, you can download and install the GradBench CLI:

```sh
cargo install --locked --git https://github.com/gradbench/gradbench
```

Then if you have [Docker][] installed, you can use the GradBench CLI to run any of our [evals](evals) against any of our [tools](tools):

```sh
gradbench run --eval 'gradbench eval hello' --tool 'gradbench tool pytorch'
```

This will first automatically download our latest nightly Docker images for the given eval and tool, and then run the eval against the tool while printing a summary of the communication log to the terminal. To save the full log to a file, use the `--output` flag. Or, to see a list of all possible subcommands:

```sh
gradbench --help
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
{ "elapsed": { "nanoseconds": 350000 }, "response": { "id": 2, "output": 2.71828, "timings": [{ "name": "evaluate", "nanoseconds": 45678 }] } }
{ "elapsed": { "nanoseconds": 400000 }, "message": { "id": 3, "kind": "analysis", "of": 2, "valid": false, "message": "Expected tau, got e." } }
{ "elapsed": { "nanoseconds": 450000 }, "response": { "id": 3 } }
{ "elapsed": { "nanoseconds": 500000 }, "message": { "id": 4, "kind": "evaluate", "module": "foo", "function": "baz", "input": { "mynumber": 121 } } }
{ "elapsed": { "nanoseconds": 550000 }, "response": { "id": 4, "output": { "yournumber": 342 }, "timings": [{ "name": "evaluate", "nanoseconds": 23456 }] } }
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
  { "id": 2, "output": 2.71828, "timings": [{ "name": "evaluate", "nanoseconds": 45678 }] }
  { "id": 3 }
  { "id": 4, "output": { "yournumber": 342 }, "timings": [{ "name": "evaluate", "nanoseconds": 23456 }] }
  { "id": 5 }
  ```

As shown by this example, the intermediary forwards every message from the eval to the tool, and vice versa. The tool only provides substantive responses for `"define"` and `"evaluate"` messages; for all others, it simply gives a response acknowledging the `"id"` of the original message.

### Specification

The session proceeds over a series of _rounds_, driven by the eval. In each round, the eval sends a _message_ with a unique `"id"`, and the tool sends a _response_ with that same `"id"`. The message also includes a `"kind"`, which has four possibilities:

1. `"kind": "start"` - the eval always sends this message first, waiting for the tool's response to ensure that it is ready to receive further messages.

2. `"kind": "define"` - the eval provides the name of a `"module"` which the tool will need in order to proceed further with this particular benchmark. This will allow the tool to respond saying whether or not it knows of and has an implementation for the module of that name.

   - The tool responds with the `"id"` and either `"success": true` or `"success": false`. In the former case, the benchmark proceeds normally. In the latter case, the tool is indicating that it does not have an implementation for the requested module, and the eval should stop and not send any further messages.

3. `"kind": "evaluate"` - the eval again provides a `"module"` name, as well as the name of a `"function"` in that module. Currently there is no formal process for registering module names or specifying the functions available in those modules; those are specified informally via documentation in the evals themselves. An `"input"` to that function is also provided; the tool will be expected to evaluate that function at that input, and return the result. Optionally, the eval may also provide a short human-readable `"description"` of the input.

   - The tool responds with the `"id"` and its `"output"` from evaluating the requested function with the given input. Optionally, the tool may also provide a list of `"timings"` for subtasks of the computation it performed. Each timing must include a `"name"` that does not need to be unique, and a number of `"nanoseconds"`. Currently, most tools only provide one entry in `"timings"`: an `"evaluate"` entry, which by convention means the amount of time that tool spent evaluating the function itself, not including other time such as JSON encoding/decoding.

4. `"kind": "analysis"` - the eval provides the ID of a prior `"evaluate"` message it performed analysis `"of"`, along with a boolean saying whether the tool's output was `"valid"`. If the output was invalid, the eval can also provide a string `"message"` explaining why.

If the tool receives any message whose `"kind"` is neither `"define"` nor `"evaluate"`, it must always respond, but does not need to include anything other than the `"id"`.

### Types

Here is a somewhat more formal description of the protocol using [TypeScript][] types.

```typescript
interface Base {
  id: string;
}

interface Duration {
  nanoseconds: number;
}

interface Timing extends Duration {
  name: string;
}

interface StartMessage extends Base {
  kind: "start";
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
  valid: boolean;
  message?: string;
}

type Message = StartMessage | DefineMessage | EvaluateMessage | AnalysisMessage;

interface DefineResponse extends Base {
  success: boolean;
  error?: string;
}

interface EvaluateResponse extends Base {
  output: any;
  timings?: Timing[];
  error?: string;
}

type Response = Base | DefineResponse | EvaluateResponse;

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

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

GradBench is licensed under the [MIT License](LICENSE).

[docker]: https://docs.docker.com/engine/install/
[json]: https://json.org/
[rust]: https://www.rust-lang.org/tools/install
[typescript]: https://www.typescriptlang.org/

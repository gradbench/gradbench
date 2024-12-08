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

GradBench decouples benchmarks from tools via a [JSON][]-based protocol. In this protocol, there is an _intermediary_ (our `run.py` script), an _eval_, and a _tool_. The eval and the tool communicate with each other by sending and receiving messages over stdout and stdin, which are intercepted by the intermediary and forwarded as appropriate.

### Example

To illustrate, here is a hypothetical example of a complete session of the protocol, as captured and reported by the intermediary:

```json
[
  {
    "message": { "id": 0, "kind": "define", "module": "foo" },
    "nanoseconds": 12345,
    "response": { "id": 0, "success": true }
  },
  {
    "message": { "id": 1, "kind": "evaluate", "module": "foo", "name": "bar", "input": 3.14159 },
    "nanoseconds": 56789,
    "response": { "id": 1, "output": 2.71828, "nanoseconds": { "evaluate": 45678 } },
    "analysis": { "id": 1, "kind": "analysis", "correct": false, "error": "Expected tau, got e." }
  },
  {
    "message": { "id": 2, "kind": "evaluate", "module": "foo", "name": "baz", "input": { "mynumber": 121 } },
    "nanoseconds": 34567,
    "response": { "id": 2, "output": { "yournumber": 342 }, "nanoseconds": { "evaluate": 23456 } },
    "analysis": { "id": 2, "kind": "analysis", "correct": true }
  },
  {
    "message": { "id": 3, "kind": "end", "validations": [ { "id": 1, "correct": false, "error": "Expected tau, got e." }, { "id": 2, "correct": true } ] }
  }
]
```

Here is that example from the perspective of the eval. (Output is listed first here, because the eval drives the protocol.)

- Output:
  ```json
  { "id": 0, "kind": "define", "module": "foo" }
  { "id": 1, "kind": "evaluate", "module": "foo", "name": "bar", "input": 3.14159 }
  { "id": 1, "kind": "analysis", "correct": false, "error": "Expected tau, got e." }
  { "id": 2, "kind": "evaluate", "module": "foo", "name": "baz", "input": { "mynumber": 121 } }
  { "id": 2, "kind": "analysis", "correct": true }
  { "id": 3, "kind": "end", "validations": [ { "id": 1, "correct": false, "error": "Expected tau, got e." }, { "id": 2, "correct": true } ] }
  ```
- Input:
  ```json
  { "id": 0, "success": true }
  { "id": 1, "output": 2.71828, "nanoseconds": { "evaluate": 45678 } }
  { "id": 2, "output": { "yournumber": 342 }, "nanoseconds": { "evaluate": 23456 } }
  ```

And here is that example from the perspective of the tool. (Input is listed first here, because the tool does not drive the protocol.)

- Input:
  ```json
  { "id": 0, "kind": "define", "module": "foo" }
  { "id": 1, "kind": "evaluate", "module": "foo", "name": "bar", "input": 3.14159 }
  { "id": 2, "kind": "evaluate", "module": "foo", "name": "baz", "input": { "mynumber": 121 } }
  ```
- Output:
  ```json
  { "id": 0, "success": true }
  { "id": 1, "output": 2.71828, "nanoseconds": { "evaluate": 45678 } }
  { "id": 2, "output": { "yournumber": 342 }, "nanoseconds": { "evaluate": 23456 } }
  ```

As shown by this example, the intermediary forwards every message from the tool back to the eval, but it only forwards `"define"` and `"evaluate"` messages from the eval to the tool.

### Specification

The session proceeds over a series of _rounds_, driven by the eval. At the beginning of each round, the eval sends a message, which always includes an `"id"` and a `"kind"`, the latter of which has three possibilities:

1. `"kind": "define"` - The eval provides the name of a `"module"` which the tool will need in order to proceed further with this particular benchmark. This will allow the tool to respond saying whether or not it knows of and has an implementation for the module of that name.

   - The intermediary forwards this message to the tool, which must respond with the same `"id"` as the original message, and either `"success": true` or `"success": false`. In the former case, the benchmark proceeds normally. In the latter case, the tool is indicating that it does not have an implementation for the requested module, and the eval should stop and not send any further messages.

2. `"kind": "evaluate"` - the eval again provides a `"module"` name, as well as the `"name"` of a function in that module. Currently there is no formal process for registering module names or specifying the functions available in those modules; those are specified informally via documentation in the evals themselves. An `"input"` to that function is also provided; the tool will be expected to evaluate that function at that input, and return the result.

   - The intermediary forwards this message to the tool, which must again respond with the same `"id"` as the original message, along with the `"output"` of evaluating the requested function with the given input. The intermediary can time this entire interaction, but that measurement is very coarse; the tool can provide more fine-grained timing data for sub-tasks in the `"nanoseconds"` field. Currently, most tools only provide one entry in `"nanoseconds"`: the `"evaluate"` entry, which by convention means the amount of time that tool spent evaluating the function itself, not including other time such as JSON encoding/decoding.

   - The intermediary forwards that response back to the eval, which can then respond back to the intermediary with a `"kind": "analysis"` message, only valid in this specific context. The `"id"` must again match that of the original message. The `"correct"` field is a boolean saying whether the tool's output was acceptable; if `"correct": false`, the eval can also provide an `"error"` field with a string message explaining why the tool's output was not acceptable.

3. `"kind": "end"`, which must be the last message of the session and is not forwarded to the tool. The eval provides a list of `"validations"`; currently these are just a reiteration of the data already provided by the eval's `"kind": "analysis"` messages.

### Types

Here is a somewhat more formal definition of the protocol using [TypeScript][] types.

```typescript
interface Message {
  id: string;
}

interface DefineMessage extends Message {
  kind: "define";
  module: string;
}

interface DefineResponse extends Message {
  success: boolean;
}

interface EvaluateMessage extends Message {
  kind: "evaluate";
  module: string;
  name: string;
  input: any;
}

interface EvaluateResponse extends Message {
  output: any;
  nanoseconds: Record<string, number>;
}

interface EvaluateAnalysis extends Message {
  correct: boolean;
  error?: string;
}

interface Validation extends Message {
  correct: boolean;
  error?: string;
}

interface EndMessage extends Message {
  kind: "end";
  validations: Validation[];
}

interface Define {
  message: DefineMessage;
  nanoseconds: number;
  response: DefineResponse;
}

interface Evaluate {
  message: EvaluateMessage;
  nanoseconds: number;
  response: EvaluateResponse;
  analysis?: EvaluateAnalysis;
}

interface End {
  message: EndMessage;
}

type Round = Define | Evaluate | End;

type Session = Round[];
```


## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

GradBench is licensed under the [MIT License](LICENSE).

[docker]: https://docs.docker.com/engine/install/
[json]: https://json.org/
[rust]: https://www.rust-lang.org/tools/install
[typescript]: https://www.typescriptlang.org/

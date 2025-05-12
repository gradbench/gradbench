# TensorFlow.js

[TensorFlow.js](https://www.tensorflow.org/js) is a library for machine learning
in JavaScript.

To run this outside Docker, use the following command:

```sh
node --disable-warning=ExperimentalWarning js/tensorflow/run.ts
```

That command (which is also used inside of the Docker image) runs TensorFlow.js
inside of [Node.js](https://nodejs.org); to manually run inside of a browser,
use this command instead:

```sh
bun run --filter=@gradbench/tensorflow dev
```

## Commentary

At time of writing, the latest version of TensorFlow.js is [v4.22.0][], which
[does not support double-precision floating point][precision], so all the eval
implementations for this tool use single-precision floating point instead. Keep
this in mind when comparing performance against the other tools in this repo,
all of which use double precision.

Also, currently this tool implementation uses the CPU [backend][], which is
ostensibly the slowest. For more representative performance numbers, we should
change to use a faster TensorFlow.js backend.

[backend]: https://www.tensorflow.org/js/guide/platform_environment
[precision]: https://stackoverflow.com/a/57425824
[v4.22.0]: https://www.npmjs.com/package/@tensorflow/tfjs/v/4.22.0

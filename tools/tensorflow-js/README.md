# TensorFlow.js

[TensorFlow.js](https://www.tensorflow.org/js) is a library for machine learning in JavaScript.

To run this outside Docker, use the following command:

```sh
node --disable-warning=ExperimentalWarning js/tensorflow/run.ts
```

That command (which is also used inside of the Docker image) runs TensorFlow.js inside of [Node.js](https://nodejs.org); to manually run inside of a browser, use this command instead:

```sh
bun run --filter=@gradbench/tensorflow dev
```

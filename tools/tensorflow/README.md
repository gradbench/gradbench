# TensorFlow

[TensorFlow][] is a machine learning library for the [Python][] programming
language.

## Commentary

### Multithreading

TensorFlow automatically uses all available cores, but unless the
`--multithreaded` flag is passed, the tool restricts it to a single thread. The
evals are written in a data-parallel style, but the speedup is modest, as most
of the work is memory-bound.

[python]: https://www.python.org/
[tensorflow]: https://www.tensorflow.org/

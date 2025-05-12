# JAX

[JAX][] is an automatic differentiation library for the [Python][] programming
language.

## Commentary

### Multithreading

Jax's implementation automatically takes advantage of multiple threads, and most
evals are implemented in a way that allows this to be efficient. However, unless
the `--multithreaded` flag is passed, the tool constrains the process to use
only a single core.

[jax]: http://jax.readthedocs.io/
[python]: https://www.python.org/

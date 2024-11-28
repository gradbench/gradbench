# Hello, world!

The `hello` eval is the simplest one; it first defines a module named `"hello"` and then alternates between asking the tool to evaluate one of the two functions in that module:

- The `"square"` function implements $f : \mathbb{R} \to \mathbb{R}$ given by $f(x) = x^2$, taking a single number as output and returning the square of that value as output.

- The `"double"` function is the gradient of the `"square"` function! It implements $\nabla f : \mathbb{R} \to \mathbb{R}$ given by $\nabla f(x) = 2x$, taking a single number as input, and returns twice that value as output.

# Finding Saddle Points with Gradient Descent

The benchmark uses gradient descent to compute the saddle point of a
simple function, and was originally proposed by Pearlmutter and
Siskind in the paper [Using Programming Language Theory to Make
Automatic Differentiation Sound and
Efficient](https://link.springer.com/chapter/10.1007/978-3-540-68942-3_8).

Specifically, the benchmark computes

```math
\text{min}_ {x} \text{min}_ {y} f(x,y)
```

where

```math
f : \mathbb{R}^2 \times \mathbb{R}^2 \rightarrow \mathbb{R}
```

is defined as the trivial function

```math
f(x,y) = (x_1^2 + y_1^2) - (x_2^2 + y_1^2)
```

The intent is that finding the minimum (*argmin*) is done via gradient
descent. Since this results in two nested argmins, this means the
occurrence of nested AD. Specifically, we need both the gradient of
$f$, and the gradient of $\text{min}_ {y} f(x,y)$ (with free $x$). An
implementation of this benchmark must be written using nested AD.

There are two instances of AD, and either can be implemented using
forward mode or reverse mode, yielding four combinations. An
implementation is expected to implement all four of these.

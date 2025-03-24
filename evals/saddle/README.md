# Finding Saddle Points with Gradient Descent

The benchmark uses gradient descent to compute the saddle point of a
simple function. This benchmark, along with [saddle](../saddle/), was
originally proposed by Pearlmutter and Siskind in the papers [Using
Programming Language Theory to Make Automatic Differentiation Sound
and
Efficient](https://link.springer.com/chapter/10.1007/978-3-540-68942-3_8)
and [Putting the Automatic Back into AD: Part I, What’s
Wrong](https://docs.lib.purdue.edu/ecetr/368/). The benchmark has also
been covered in [Putting the Automatic Back into AD: Part I, What’s
Wrong](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1369&context=ecetr),
which mainly discusses how the tools of the time had a very hard time
handling it correctly.

## Specification

The benchmark computes

```math
\text{min}_ {x\in\mathbb{R}^2} \text{max}_ {y\in\mathbb{R}^2} f(x,y)
```

where

```math
f : \mathbb{R}^2 \times \mathbb{R}^2 \rightarrow \mathbb{R}
```

is defined as the trivial function

```math
f(x,y) = (x_1^2 + y_1^2) - (x_2^2 + y_1^2)
```

We must produce the two points $x$ and $y$ as the results.

The intent is that finding the minimum (_argmin_) and maximum
(_argmax_), both via gradient descent. Since this results in an argmin
containing an argmax, this means the occurrence of nested AD. An
implementation of this benchmark must be written using nested AD.
There are two instances of AD, and either can be implemented using
forward mode or reverse mode, yielding four combinations. An
implementation is expected to implement all four of these.

Operationally, as input we are given a starting point $s$. This is
used as the starting point for all instances of gradient descent (both
inner and outer). The algorithm has two steps:

1. Find the point $x$ that minimises $\text{max}_ {y\in\mathbb{R}^2}
   f(x,y)$ that is, compute the argmin. This is the part that requires
   nested gradient descent, as we need to (repeatedly) compute the
   inner argmax.

2. Now that $x$ is known, find the $y$ that maximises the function
   $f(x,y)$ that is, compute the argmax, but this time not in a nested
   case.

The benchmark can be implemented in four function variants, which at
the GradBench level exist as four functions. They differ in which mode
of AD is used for step 1 above:

- `ff`, where the argmin uses forward mode and the argmax uses forward mode.
- `fr`, where the argmin uses forward mode and the argmax uses reverse mode.
- `rf`, where the argmin uses reverse mode and the argmax uses forward mode.
- `rr`, where the argmin uses reverse mode and the argmax uses reverse mode.

The argmax of step 2 must use the same mode of AD as the argmin of step 1.

All function variants accept the same input and must produce the same
results.

## Protocol

The evaluation protocol is specified in terms of [TypeScript][] types
and references [types defined in the GradBench protocol
description][protocol].

### Inputs

The eval sends a leading `DefineMessage` with a module named `saddle`,
followed by `EvaluateMessages`. The `input` field of any
`EvaluateMessage` will be an instance of the `SaddleInput` type
defined below. The `function` field will one of the strings `"ff"`,
`"fr"`, `"rr"`, `"rf"`, which all have the same input/output
interface.

```typescript
interface SaddleInput extends Runs {
  start: double[2];
}
```

Here `start` encodes the point _s_ listed in the specification.

### Outputs

A tool must respond to an `EvaluateMessage` with an
`EvaluateResponse`. The type of the `output` field in the
`EvaluateResponse` must be `SaddleOutput`.

```typescript
type SaddleOutput = double[4];
```

The output contains the points `x` and `y` concatenated.

[protocol]: /CONTRIBUTING.md#types
[typescript]: https://www.typescriptlang.org/

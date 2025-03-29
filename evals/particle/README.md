# Charged particle trajectory

The benchmark models a charged particle accelerated by an electric field formed by a pair of repulsive bodies, with the goal being to find a control parameter that causes the movement of the particle to intersect the origin. This benchmark, along with [saddle](../saddle), was originally proposed by Pearlmutter and Siskind in the paper [Using Programming Language Theory to Make Automatic Differentiation Sound and Efficient](https://link.springer.com/chapter/10.1007/978-3-540-68942-3_8). The benchmark has also been covered in [Putting the Automatic Back into AD: Part I, What’s Wrong](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1369&context=ecetr), which mainly discusses how the tools of the time had a very hard time handling it correctly.

## Specification

A particle is modeled by position $\mathbf{x}(t)$ and velocity
$\dot{\mathbf{x}}(t)$ where $t$ is the time, and the acceleration is
given by

```math
p(\mathbf{x};w) = \Vert\mathbf{x}-(10,10-w)\Vert^{-1} + \Vert\mathbf{x}-(10,0)\Vert^{-1}
```

where $w$ is a control parameter that we can adjust. The particle hits
the $x$-axis at position $\mathbf{x}(t_f)$. We desire to compute

```math
\text{min}_w\ x_0(t_f)^2
```

which makes the particle intersect the origin (this is an _argmin_
operation).

[Naive Euler ODE integration][euler] is used to compute the particle's
path, with linear interpolation to find the intersection with the $x$
axis (see paper for the formula, or the code for one of the existing
tools, this is not expressible with GitHub's math support). This ODE
to find $x_0(t_f)^2$ involves taking the gradient of the $p$ function
defined above. The _argmin_ operation is solved with gradient descent.
Thus, we have an instance of nested AD.

The benchmark can be implemented in four function variants, which at
the GradBench level exist as four functions. They differ in which mode
of AD is used for step 1 above:

- `ff`, where the argmin uses forward mode and the Euler method uses forward mode.
- `fr`, where the argmin uses forward mode and the Euler method uses reverse mode.
- `rf`, where the argmin uses reverse mode and the Euler method uses forward mode.
- `rr`, where the argmin uses reverse mode and the Euler method uses reverse mode.

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
  w: double;
}
```

This is the starting point for computing $w$.

### Outputs

A tool must respond to an `EvaluateMessage` with an
`EvaluateResponse`. The type of the `output` field in the
`EvaluateResponse` must be `SaddleOutput`.

```typescript
type SaddleOutput = double;
```

The output is the final value of $w$.

## Commentary

The challenging part about `particle` (and [saddle][]) is that it
involves nested AD, and not merely in the simple way needed by
[kmeans][] for computing hessians. These benchmarks things optimise
objective functions that themselves contain instances of AD (in the
case of 'saddle', this is a nested solver). The`'particle'`eval even
has an unbounded `while` loop in the primal function, which can be a
challenge to some tools. In particular, for the C++ tools built on
operator overloading, the types can get quite involved.

`particle` is a scalar benchmark, with no large arrays, and no
meaningful potential for parallel execution. In principle, the two
"attractors" that depend on $w$ could be extended to a much larger
quantity, but that might not actually be a sensible thing to simulate.

It is not required that all of the four function variants are
implemented by instantiating some generic function (although some of
our tools do it like that). It is fine to have four completely
independent implementations.

[protocol]: /CONTRIBUTING.md#types
[typescript]: https://www.typescriptlang.org/
[paper]: https://link.springer.com/chapter/10.1007/978-3-540-68942-3_8
[euler]: https://en.wikipedia.org/wiki/Euler_method
[kmeans]: ../kmeans
[saddle]: ../saddle

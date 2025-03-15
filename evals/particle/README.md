# Charged particle trajectory

The benchmark models a charged particle accelerated by an electric
field, with the goal being to find a control parameter that causes the
movement of the particle to intersect the origin. The benchmark and
was originally proposed by Pearlmutter and Siskind in the paper [Using
Programming Language Theory to Make Automatic Differentiation Sound
and
Efficient](https://link.springer.com/chapter/10.1007/978-3-540-68942-3_8).

Specifically, the particle is modeled by position $x(t)$ and velocity
$\dot{x}(t)$, and the acceleration is given by

```math
p(x;w) = ||x-(10,10-w)||^{-1} + ||x-(10,0)||^{-1}
```

where $w$ is a control parameter that we can adjust. The particle hits
the $x$-axis at position $x(t_f)$, and the goal is to compute

```math
\text{min}_w x_0(t_f)^2
```

which makes the particle intersect the origin.

# Julia

This directory contains a library used by the Julia-related tools. Each eval has
a module in [GradBench/src/benchmarks/](GradBench/src/benchmarks/), which
defines IO facilities and one or more reference implementations of the primal
functions.

Due to the diversity of Julia AD tools, some of the primal functions exist in
multiple versions, in order to cater to the peculiarities of the different
tools. For example, [Zygote][] cannot handle destructive updates, while
[Enzyme.jl][] often performs better using an imperative style. Each module
should document the differences between its various versions.

[Zygote]: /tools/zygote
[Enzyme.jl]: /tools/enzyme-jl

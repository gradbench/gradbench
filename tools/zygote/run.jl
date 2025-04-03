import GradBench

include("run_hello.jl")
# Zygote can't differentiate through mutation
# include("run_ode.jl")

GradBench.main("zygote")

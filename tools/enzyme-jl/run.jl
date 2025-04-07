import GradBench

include("run_hello.jl")
include("run_ode.jl")

GradBench.main("enzyme-jl")

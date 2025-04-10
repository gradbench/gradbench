import GradBench

include("run_hello.jl")
include("run_ode.jl")
include("run_lse.jl")

GradBench.main("enzyme-jl")

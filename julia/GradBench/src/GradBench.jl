module GradBench

import JSON

"""
    Experiment

Defines a measurment experiment.

## Interface
- `from_json()`
- `ret = (::Experiment)(args...)`
- Optional: `postprocess(::Experiment, ret)`
"""
abstract type Experiment end

function from_json end
function postprocess(::Experiment, ret)
    return ret
end

const DISPATCH_TABLE = Dict{String,Dict{String,Experiment}}()

function register!(mod, experiments)
    if haskey(DISPATCH_TABLE, mod)
        error("mod $mod is already registered")
    end
    DISPATCH_TABLE[mod] = experiments
    return
end

# Avoid measuring dispatch overhead
function measure(experiment::E, args...) where {E<:Experiment}
    start = time_ns()
    ret = experiment(args...)
    done = time_ns()
    return ret, done - start
end

function run(params)
    mod = DISPATCH_TABLE[params["module"]]
    experiment = mod[params["function"]]
    min_runs = get(params, "min_runs", 1)
    min_seconds = get(params, "min_seconds", 0)
    @assert min_runs > 0

    args = from_json(experiment, params["input"])
    timings = Any[]

    # Measure
    elapsed_seconds = 0
    i = 1
    ret = nothing
    while i <= min_runs || elapsed_seconds <= min_seconds
        ret, t = measure(experiment, args...)
        push!(timings, Dict("name" => "evaluate", "nanoseconds" => t))
        elapsed_seconds += t / 1e9
        i += 1
    end
    @assert ret !== nothing

    return Dict("success" => true, "output" => postprocess(experiment, ret), "timings" => timings)
end

function main(tool)
    while !eof(stdin)
        message = JSON.parse(readline(stdin))
        response = Dict()
        if message["kind"] == "start"
            response["tool"] = tool
        elseif message["kind"] == "evaluate"
            response = run(message)
        elseif message["kind"] == "define"
            response["success"] = haskey(DISPATCH_TABLE, message["module"])
        end
        response["id"] = message["id"]
        println(JSON.json(response))
    end
    return
end

include("benchmarks/ba.jl")
include("benchmarks/det.jl")
include("benchmarks/gmm.jl")
include("benchmarks/hello.jl")
include("benchmarks/ht.jl")
include("benchmarks/ode.jl")
include("benchmarks/lse.jl")
include("benchmarks/lstm.jl")

end # module GradBench

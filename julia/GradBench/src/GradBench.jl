module GradBench

import JSON

const DISPATCH_TABLE = Dict{String,Dict{String,Any}}()

function register!(mod, functions)
    if haskey(DISPATCH_TABLE, mod)
        error("mod $mod is already registered")
    end
    DISPATCH_TABLE[mod] = functions
    return
end

# Avoid measuring dispatch overhead
function measure(func::F, arg) where {F}
    start = time_ns()
    ret = func(arg)
    done = time_ns()
    return ret, done - start
end

function run(params)
    mod = DISPATCH_TABLE[params["module"]]
    func = mod[params["function"]]
    arg = params["input"]
    min_runs = get(params, "min_runs", 1)
    min_seconds = get(params, "min_seconds", 0)
    @assert min_runs > 0

    # TODO: Prepare (parse JSON?)
    # TODO: pre-allocate output?
    timings = Any[]

    # Measure
    elapsed_seconds = 0
    i = 1
    ret = nothing
    while i <= min_runs || elapsed_seconds <= min_seconds
        ret, t = measure(func, arg)
        push!(timings, Dict("name" => "evaluate", "nanoseconds" => t))
        elapsed_seconds += t / 1e9
        i += 1
    end
    @assert ret !== nothing
    return Dict("success" => true, "output" => ret, "timings" => timings)
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

include("benchmarks/gmm.jl")
include("benchmarks/hello.jl")
include("benchmarks/ode.jl")
include("benchmarks/lse.jl")
include("benchmarks/lstm.jl")

end # module GradBench

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
    ret, t = measure(func, arg)
    timings = [Dict("name" => "evaluate", "nanoseconds" => t)]
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

include("benchmarks/hello.jl")
include("benchmarks/ode.jl")

end # module GradBench

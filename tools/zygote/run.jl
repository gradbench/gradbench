module Functions

import Zygote

function square(x)
  return x * x
end

function double(x)
  z, = Zygote.gradient(square, x)
  return z
end

end

import JSON

function resolve(name)
  return getfield(Functions, Symbol(name))
end

function run(params)
  func = resolve(params["name"])
  arg = params["input"]
  start = time_ns()
  ret = func(arg)
  done = time_ns()
  ns = Dict("evaluate" => done - start)
  return Dict("output" => ret, "nanoseconds" => ns)
end

function main()
  while !eof(stdin)
    message = JSON.parse(readline(stdin))
    response = Dict()
    if message["kind"] == "evaluate"
      response = run(message)
    end
    response["id"] = message["id"]
    println(JSON.json(response))
  end
end

main()

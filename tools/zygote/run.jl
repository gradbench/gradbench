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
  vals = [arg["value"] for arg in params["arguments"]]
  start = time_ns()
  ret = func(vals...)
  done = time_ns()
  return Dict("return" => ret, "nanoseconds" => done - start)
end

function main()
  cfg = JSON.parse(read(stdin, String))
  outputs = map(run, cfg["inputs"])
  println(JSON.json(Dict("outputs" => outputs)))
end

main()

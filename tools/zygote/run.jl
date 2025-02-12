module hello

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

function resolve(mod, name)
  module_sym = Symbol(mod)
  eval(:(using .$module_sym)) #imports module
  mod_ref = getfield(Main, module_sym) #retrieves module
  name_sym = Symbol(name)
  return getfield(mod_ref, name_sym)
end

function run(params)
  func = resolve(params["module"], params["function"])
  arg = params["input"]
  start = time_ns()
  ret = func(arg)
  done = time_ns()
  timings = [Dict("name" => "evaluate", "nanoseconds" => done - start)]
  return Dict("success" => true, "output" => ret, "timings" => timings)
end

function main()
  while !eof(stdin)
    message = JSON.parse(readline(stdin))
    response = Dict()
    if message["kind"] == "evaluate"
      response = run(message)
    elseif message["kind"] == "define"
      success = true
      try
        module_sym = Symbol(message["module"])
        eval(:(using .$module_sym))
        getfield(Main, module_sym)
      catch
        success = false
      end
      response["success"] = success
    end
    response["id"] = message["id"]
    println(JSON.json(response))
  end
end

main()

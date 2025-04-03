module Hello

function square(x)
    return x * x
end

precompile(square, (Float64,))

end

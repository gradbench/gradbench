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

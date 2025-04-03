import GradBench

module hello

import GradBench.hello: square
import Zygote

function double(x)
  z, = Zygote.gradient(square, x)
  return z
end

end

GradBench.main("zygote")

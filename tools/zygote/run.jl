import Gradbench

module hello

import Gradbench.hello: square
import Zygote

function double(x)
  z, = Zygote.gradient(square, x)
  return z
end

end

Gradbench.main()

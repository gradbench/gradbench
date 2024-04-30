import Zygote

function least_squares(; x, y)
  x = hcat(ones(size(y)), x)
  return function (beta)
    epsilon = y - x * beta
    return epsilon' * epsilon
  end
end

function linear_regression(; x, y, eta)
  f = least_squares(; x, y)
  b = zeros(1 + size(x, 2))
  while true
    grad, = Zygote.gradient(f, b)
    b1 = b - eta * grad
    if b1 == b
      break
    end
    b = b1
  end
  return b
end

function main()
  beta = linear_regression(; eta=1e-4,
    x=[10; 8; 13; 9; 11; 14; 6; 4; 12; 7; 5],
    y=[8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
  )
  println(beta)
end

main()

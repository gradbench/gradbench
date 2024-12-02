entry square (x: f64) = x * x

entry double (x: f64) = vjp square x 1

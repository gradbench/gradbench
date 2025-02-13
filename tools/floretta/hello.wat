(module
  (func (export "square") (param f64) (result f64)
    (f64.mul
      (local.get 0)
      (local.get 0))))

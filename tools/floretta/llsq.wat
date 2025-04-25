(module
  (memory (export "memory") 0)
  (func (export "llsq")
    (param $n i32) (param $m i32) (param $x i32) (result f64)
    (local $sum f64) (local $i i32) (local $z f64) (local $t f64) (local $j i32) (local $c f64)
    (loop $outer
      (if
        (i32.lt_s
          (local.get $i)
          (local.get $n))
        (then
          (local.set $t
            (f64.sub
              (f64.div
                (f64.mul
                  (f64.convert_i32_s
                    (local.get $i))
                  (f64.const 2))
                (f64.sub
                  (f64.convert_i32_s
                    (local.get $n))
                  (f64.const 1)))
              (f64.const 1)))
          (local.set $z
            (f64.copysign
              (f64.const 1)
              (local.get $t)))
          (local.set $j
            (i32.const 0))
          (local.set $c
            (f64.const 1))
          (loop $inner
            (if
              (i32.lt_s
                (local.get $j)
                (local.get $m))
              (then
                (local.set $z
                  (f64.sub
                    (local.get $z)
                    (f64.mul
                      (f64.load
                        (i32.add
                          (local.get $x)
                          (i32.mul
                            (i32.const 8)
                            (local.get $j))))
                      (local.get $c))))
                (local.set $c
                  (f64.mul
                    (local.get $c)
                    (local.get $t)))
                (local.set $j
                  (i32.add
                    (local.get $j)
                    (i32.const 1)))
                (br $inner))))
          (local.set $sum
            (f64.add
              (local.get $sum)
              (f64.mul
                (local.get $z)
                (local.get $z))))
          (local.set $i
            (i32.add
              (local.get $i)
              (i32.const 1)))
          (br $outer))))
    (f64.div
      (local.get $sum)
      (f64.const 2))))

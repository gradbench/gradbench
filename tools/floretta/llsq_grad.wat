(module
  (memory (export "memory") 0)
  (memory $bwd (export "memory_bwd") 0)
  (memory $z 3 3)
  (memory $c 257 257)
  (global $n (mut i32)
    (i32.const 0))
  (global $m (mut i32)
    (i32.const 0))
  (func (export "llsq")
    (param $n i32) (param $m i32) (param $x i32) (result f64)
    (local $sum f64) (local $i i32) (local $z f64) (local $t f64) (local $j i32) (local $c f64)
    (global.set $n
      (local.get $n))
    (global.set $m
      (local.get $m))
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
                (f64.store $c
                  (i32.mul
                    (i32.const 8)
                    (i32.add
                      (i32.mul
                        (global.get $m)
                        (local.get $i))
                      (local.get $j)))
                  (local.get $c))
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
          (f64.store $z
            (i32.mul
              (i32.const 8)
              (local.get $i))
            (local.get $z))
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
      (f64.const 2)))
  (func (export "backprop") (param $dy f64)
    (local $dsum f64) (local $i i32) (local $dz f64) (local $j i32)
    (local.set $dsum
      (f64.div
        (local.get $dy)
        (f64.const 2)))
    (local.set $i
      (i32.sub
        (global.get $n)
        (i32.const 1)))
    (loop $outer
      (if
        (i32.ge_s
          (local.get $i)
          (i32.const 0))
        (then
          (local.set $j
            (i32.sub
              (global.get $m)
              (i32.const 1)))
          (local.set $dz
            (f64.mul
              (f64.mul
                (f64.const 2)
                (f64.load $z
                  (i32.mul
                    (i32.const 8)
                    (local.get $i))))
              (local.get $dsum)))
          (loop $inner
            (if
              (i32.ge_s
                (local.get $j)
                (i32.const 0))
              (then
                (f64.store $bwd
                  (i32.mul
                    (i32.const 8)
                    (local.get $j))
                  (f64.add
                    (f64.load $bwd
                      (i32.mul
                        (i32.const 8)
                        (local.get $j)))
                    (f64.mul
                      (f64.neg
                        (local.get $dz))
                      (f64.load $c
                        (i32.mul
                          (i32.const 8)
                          (i32.add
                            (i32.mul
                              (global.get $m)
                              (local.get $i))
                            (local.get $j)))))))
                (local.set $j
                  (i32.sub
                    (local.get $j)
                    (i32.const 1)))
                (br $inner))))
          (local.set $i
            (i32.sub
              (local.get $i)
              (i32.const 1)))
          (br $outer))))))

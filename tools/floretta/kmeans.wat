(module
  (memory (export "memory") 0)
  (func $square (param f64) (result f64)
    (f64.mul
      (local.get 0)
      (local.get 0)))
  (func $dist (param $d i32) (param $C_j i32) (param $P_i i32) (result f64)
    (local $sum f64) (local $l i32)
    (loop $continue
      (if
        (i32.lt_s
          (local.get $l)
          (local.get $d))
        (then
          (local.set $sum
            (f64.add
              (local.get $sum)
              (call $square
                (f64.sub
                  (f64.load
                    (i32.add
                      (local.get $C_j)
                      (i32.mul
                        (i32.const 8)
                        (local.get $l))))
                  (f64.load
                    (i32.add
                      (local.get $P_i)
                      (i32.mul
                        (i32.const 8)
                        (local.get $l))))))))
          (local.set $l
            (i32.add
              (local.get $l)
              (i32.const 1)))
          (br $continue))))
    (local.get $sum))
  (func $min_dist
    (param $d i32) (param $k i32) (param $C i32) (param $P_i i32) (result f64)
    (local $min f64) (local $j i32)
    (local.set $min
      (f64.const inf))
    (loop $continue
      (if
        (i32.lt_s
          (local.get $j)
          (local.get $k))
        (then
          (local.set $min
            (f64.min
              (local.get $min)
              (call $dist
                (local.get $d)
                (i32.add
                  (local.get $C)
                  (i32.mul
                    (i32.mul
                      (i32.const 8)
                      (local.get $d))
                    (local.get $j)))
                (local.get $P_i))))
          (local.set $j
            (i32.add
              (local.get $j)
              (i32.const 1)))
          (br $continue))))
    (local.get $min))
  (func (export "cost")
    (param $d i32) (param $k i32) (param $n i32) (param $C i32) (param $P i32) (result f64)
    (local $sum f64) (local $i i32)
    (loop $continue
      (if
        (i32.lt_s
          (local.get $i)
          (local.get $n))
        (then
          (local.set $sum
            (f64.add
              (local.get $sum)
              (call $min_dist
                (local.get $d)
                (local.get $k)
                (local.get $C)
                (i32.add
                  (local.get $P)
                  (i32.mul
                    (i32.mul
                      (i32.const 8)
                      (local.get $d))
                    (local.get $i))))))
          (local.set $i
            (i32.add
              (local.get $i)
              (i32.const 1)))
          (br $continue))))
    (local.get $sum)))

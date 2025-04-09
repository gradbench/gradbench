(module
  (import "math" "exp" (func $exp (param f64) (result f64)))
  (memory (export "memory") 0)
  (func (export "logsumexp") (param $n i32) (param $x i32) (result f64)
    (local $i i32) (local $a f64) (local $lse f64)
    (local.set $a
      (f64.const -inf))
    (loop $max
      (if
        (i32.lt_s
          (local.get $i)
          (local.get $n))
        (then
          (local.set $a
            (f64.max
              (local.get $a)
              (f64.load
                (i32.add
                  (local.get $x)
                  (i32.mul
                    (i32.const 8)
                    (local.get $i))))))
          (br $max))))
    (loop $sum
      (if
        (i32.lt_s
          (local.get $i)
          (local.get $n))
        (then
          (local.set $lse
            (f64.add
              (local.get $lse)
              (call $exp
                (f64.sub
                  (f64.load
                    (i32.add
                      (local.get $x)
                      (i32.mul
                        (i32.const 8)
                        (local.get $i))))
                  (local.get $a)))))
          (br $sum))))
    (local.get $lse)))

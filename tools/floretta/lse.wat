(module
  (import "math" "exp" (func $exp (param f64) (result f64)))
  (import "math" "log" (func $log (param f64) (result f64)))
  (memory (export "memory") 0)
  (func $maximum (param $n i32) (param $x i32) (result f64)
    (local $a f64) (local $i i32)
    (local.set $a
      (f64.const -inf))
    (loop $continue
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
          (local.set $i
            (i32.add
              (local.get $i)
              (i32.const 1)))
          (br $continue))))
    (local.get $a))
  (func (export "logsumexp") (param $n i32) (param $x i32) (result f64)
    (local $sum f64) (local $a f64) (local $i i32)
    (local.set $a
      (call $maximum
        (local.get $n)
        (local.get $x)))
    (loop $continue
      (if
        (i32.lt_s
          (local.get $i)
          (local.get $n))
        (then
          (local.set $sum
            (f64.add
              (local.get $sum)
              (call $exp
                (f64.sub
                  (f64.load
                    (i32.add
                      (local.get $x)
                      (i32.mul
                        (i32.const 8)
                        (local.get $i))))
                  (local.get $a)))))
          (local.set $i
            (i32.add
              (local.get $i)
              (i32.const 1)))
          (br $continue))))
    (f64.add
      (local.get $a)
      (call $log
        (local.get $sum)))))

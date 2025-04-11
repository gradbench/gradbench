(module
  (import "math" "exp" (func $exp (param f64) (result f64)))
  (import "math" "log" (func $log (param f64) (result f64)))
  (memory (export "memory") 0)
  (memory $bwd (export "memory_bwd") 0)
  (memory $tape1 20)
  (memory $tape8 157)
  (global $n (mut i32)
    (i32.const 0))
  (global $a (mut f64)
    (f64.const -inf))
  (global $sum (mut f64)
    (f64.const 0))
  (func $maximum (param $n i32) (param $x i32) (result f64)
    (local $a f64) (local $i i32) (local $b f64)
    (local.set $a
      (f64.const -inf))
    (loop $continue
      (if
        (i32.lt_s
          (local.get $i)
          (local.get $n))
        (then
          (local.set $b
            (f64.load
              (i32.add
                (local.get $x)
                (i32.mul
                  (i32.const 8)
                  (local.get $i)))))
          (if
            (f64.ge
              (local.get $b)
              (local.get $a))
            (then
              (i32.store8 $tape1
                (local.get $i)
                (i32.const 1)))
            (else
              (i32.store8 $tape1
                (local.get $i)
                (i32.const 0))))
          (local.set $a
            (f64.max
              (local.get $a)
              (local.get $b)))
          (local.set $i
            (i32.add
              (local.get $i)
              (i32.const 1)))
          (br $continue))))
    (local.get $a))
  (func $maximum_bwd (param $da f64) (local $i i32) (local $p i32)
    (local.set $i
      (i32.sub
        (global.get $n)
        (i32.const 1)))
    (loop $continue
      (if
        (i32.ge_s
          (local.get $i)
          (i32.const 0))
        (then
          (if
            (i32.load8_u $tape1
              (local.get $i))
            (then
              (local.set $p
                (i32.mul
                  (i32.const 8)
                  (local.get $i)))
              (f64.store
                (local.get $p)
                (f64.add
                  (local.get $da)
                  (f64.load $bwd
                    (local.get $p))))
              (return)))
          (local.set $i
            (i32.sub
              (local.get $i)
              (i32.const 1)))
          (br $continue)))))
  (func (export "logsumexp") (param $n i32) (param $x i32) (result f64)
    (local $sum f64) (local $a f64) (local $i i32) (local $p i32) (local $z f64)
    (global.set $n
      (local.get $n))
    (if
      (i32.ne
        (local.get $x)
        (i32.const 0))
      (then
        (unreachable)))
    (local.set $a
      (call $maximum
        (local.get $n)
        (local.get $x)))
    (global.set $a
      (local.get $a))
    (loop $continue
      (if
        (i32.lt_s
          (local.get $i)
          (local.get $n))
        (then
          (local.set $p
            (i32.add
              (local.get $x)
              (i32.mul
                (i32.const 8)
                (local.get $i))))
          (local.set $z
            (call $exp
              (f64.sub
                (f64.load
                  (local.get $p))
                (local.get $a))))
          (f64.store $tape8
            (local.get $p)
            (local.get $z))
          (local.set $sum
            (f64.add
              (local.get $sum)
              (local.get $z)))
          (local.set $i
            (i32.add
              (local.get $i)
              (i32.const 1)))
          (br $continue))))
    (global.set $sum
      (local.get $sum))
    (f64.add
      (local.get $a)
      (call $log
        (local.get $sum))))
  (func (export "backprop") (param $dy f64)
    (local $da f64) (local $i i32) (local $p i32) (local $dz f64) (local $dw f64)
    (local.set $da
      (local.get $dy))
    (local.set $i
      (i32.sub
        (global.get $n)
        (i32.const 1)))
    (loop $continue
      (if
        (i32.ge_s
          (local.get $i)
          (i32.const 0))
        (then
          (local.set $p
            (i32.mul
              (i32.const 8)
              (local.get $i)))
          (local.set $dz
            (f64.div
              (local.get $dy)
              (global.get $sum)))
          (local.set $dw
            (f64.mul
              (local.get $dz)
              (f64.load $tape8
                (local.get $p))))
          (f64.store $bwd
            (local.get $p)
            (f64.add
              (local.get $dw)
              (f64.load $bwd
                (local.get $p))))
          (local.set $da
            (f64.sub
              (local.get $da)
              (local.get $dw)))
          (local.set $i
            (i32.sub
              (local.get $i)
              (i32.const 1)))
          (br $continue))))
    (call $maximum_bwd
      (local.get $da))))

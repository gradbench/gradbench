(module
  (import "math" "exp" (func $exp (param f64) (result f64)))
  (import "math" "log" (func $log (param f64) (result f64)))
  (import "math" "multigammaln" (func $multigammaln (param i32) (param f64) (result f64)))
  (memory (export "memory") 0)
  (global $free8 (export "free8") (mut i32)
    (i32.const 0))

  (func $alloc8 (export "malloc") (param $words i32) (result i32)
    (local $free8 i32) (local $pages i32)
    (i32.shl
      (local.tee $free8
        (global.get $free8))
      (i32.const 3))
    (if
      (local.tee $pages
        (i32.sub
          (i32.shr_u
            (i32.add
              (i32.add
                (local.get $free8)
                (local.get $words))
              (i32.const 8191))
            (i32.const 13))
          (memory.size)))
      (then
        (drop
          (memory.grow
            (local.get $pages)))))
    (global.set $free8
      (i32.add
        (local.get $free8)
        (local.get $words))))

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

  (func $logsumexp (param $n i32) (param $x i32) (result f64)
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
        (local.get $sum))))

  (func $quadratic_form_element
    (param $mu i32) (param $r i32) (param $l i32) (param $x i32) (param $i i32) (result f64)
    (local $k i32) (local $e f64) (local $j i32)
    (local.set $k
      (i32.div_s
        (i32.mul
          (local.get $i)
          (i32.sub
            (local.get $i)
            (i32.const 1)))
        (i32.const 2)))
    (loop
      (if
        (i32.lt_s
          (local.get $j)
          (local.get $i))
        (then
          (local.set $e
            (f64.add
              (local.get $e)
              (f64.mul
                (f64.load
                  (i32.add
                    (local.get $l)
                    (i32.mul
                      (i32.const 8)
                      (i32.add
                        (local.get $k)
                        (local.get $j)))))
                (f64.sub
                  (f64.load
                    (i32.add
                      (local.get $x)
                      (i32.mul
                        (i32.const 8)
                        (local.get $j))))
                  (f64.load
                    (i32.add
                      (local.get $mu)
                      (i32.mul
                        (i32.const 8)
                        (local.get $j))))))))
          (local.set $j
            (i32.add
              (local.get $j)
              (i32.const 1)))
          (br 1))))
    (f64.add
      (local.get $e)
      (f64.mul
        (f64.load
          (i32.add
            (local.get $r)
            (i32.mul
              (i32.const 8)
              (local.get $i))))
        (f64.sub
          (f64.load
            (i32.add
              (local.get $x)
              (i32.mul
                (i32.const 8)
                (local.get $i))))
          (f64.load
            (i32.add
              (local.get $mu)
              (i32.mul
                (i32.const 8)
                (local.get $i))))))))

  (func (export "gmm")
    (param $D i32) (param $K i32) (param $N i32) (param $x i32) (param $m i32) (param $gamma f64) (param $alpha i32) (param $mu i32) (param $q i32) (param $l i32) (result f64)
    (local $s i32) (local $r i32) (local $k i32) (local $j i32) (local $log_likelihood f64) (local $i i32) (local $x_i i32) (local $beta i32) (local $mu_k i32) (local $r_k i32) (local $l_k i32) (local $normsq f64) (local $sum f64) (local $tmp f64) (local $frobenius f64) (local $sum_q f64) (local $n i32) (local $log_prior f64)

    (local.set $s
      (i32.div_s
        (i32.mul
          (local.get $D)
          (i32.sub
            (local.get $D)
            (i32.const 1)))
        (i32.const 2)))

    (local.set $r
      (call $alloc8
        (i32.mul
          (local.get $K)
          (local.get $D))))
    (local.set $k
      (i32.const 0))
    (loop
      (if
        (i32.lt_s
          (local.get $k)
          (local.get $K))
        (then
          (local.set $j
            (i32.const 0))
          (loop
            (if
              (i32.lt_s
                (local.get $j)
                (local.get $D))
              (then
                (f64.store
                  (i32.add
                    (local.get $r)
                    (i32.mul
                      (i32.const 8)
                      (i32.add
                        (i32.mul
                          (local.get $k)
                          (local.get $D))
                        (local.get $j))))
                  (call $exp
                    (f64.load
                      (i32.add
                        (local.get $q)
                        (i32.mul
                          (i32.const 8)
                          (i32.add
                            (i32.mul
                              (local.get $k)
                              (local.get $D))
                            (local.get $j)))))))
                (local.set $j
                  (i32.add
                    (local.get $j)
                    (i32.const 1)))
                (br 1))))
          (local.set $k
            (i32.add
              (local.get $k)
              (i32.const 1)))
          (br 1))))

    (local.set $log_likelihood
      (f64.neg
        (f64.mul
          (f64.convert_i32_s
            (local.get $N))
          (f64.add
            (f64.mul
              (f64.div
                (f64.convert_i32_s
                  (local.get $D))
                (f64.const 2))
              (call $log
                (f64.const 6.283185307179586)))
            (call $logsumexp
              (local.get $K)
              (local.get $alpha))))))
    (local.set $i
      (i32.const 0))
    (loop
      (if
        (i32.lt_s
          (local.get $i)
          (local.get $N))
        (then
          (local.set $x_i
            (i32.add
              (local.get $x)
              (i32.mul
                (i32.const 8)
                (i32.mul
                  (local.get $i)
                  (local.get $D)))))
          (local.set $beta
            (call $alloc8
              (local.get $K)))
          (local.set $k
            (i32.const 0))
          (loop
            (if
              (i32.lt_s
                (local.get $k)
                (local.get $K))
              (then
                (local.set $mu_k
                  (i32.add
                    (local.get $mu
                      (i32.mul
                        (i32.const 8)
                        (i32.mul
                          (local.get $k)
                          (local.get $D))))))
                (local.set $r_k
                  (i32.add
                    (local.get $r
                      (i32.mul
                        (i32.const 8)
                        (i32.mul
                          (local.get $k)
                          (local.get $D))))))
                (local.set $l_k
                  (i32.add
                    (local.get $l
                      (i32.mul
                        (i32.const 8)
                        (i32.mul
                          (local.get $k)
                          (local.get $s))))))
                (local.set $normsq
                  (f64.const 0))
                (local.set $sum
                  (f64.const 0))
                (local.set $j
                  (i32.const 0))
                (loop
                  (if
                    (i32.lt_s
                      (local.get $j)
                      (local.get $D))
                    (then
                      (local.set $normsq
                        (f64.add
                          (local.get $normsq)
                          (f64.mul
                            (local.tee $tmp
                              (call $quadratic_form_element
                                (local.get $mu_k)
                                (local.get $r_k)
                                (local.get $l_k)
                                (local.get $x_i)
                                (local.get $j)))
                            (local.get $tmp))))
                      (local.set $sum
                        (f64.add
                          (local.get $sum)
                          (f64.load
                            (i32.add
                              (local.get $q)
                              (i32.mul
                                (i32.const 8)
                                (i32.add
                                  (i32.mul
                                    (local.get $k)
                                    (local.get $D))
                                  (local.get $j)))))))
                      (local.set $j
                        (i32.add
                          (local.get $j)
                          (i32.const 1)))
                      (br 1))))
                (f64.store
                  (i32.add
                    (local.get $beta)
                    (i32.mul
                      (i32.const 8)
                      (local.get $k)))
                  (f64.add
                    (f64.sub
                      (f64.load
                        (i32.add
                          (local.get $alpha)
                          (i32.mul
                            (i32.const 8)
                            (local.get $k))))
                      (f64.div
                        (local.get $normsq)
                        (f64.const 2)))
                    (local.get $sum)))
                (local.set $k
                  (i32.add
                    (local.get $k)
                    (i32.const 1)))
                (br 1))))
          (local.set $log_likelihood
            (f64.add
              (local.get $log_likelihood)
              (call $logsumexp
                (local.get $K)
                (local.get $beta))))
          (local.set $i
            (i32.add
              (local.get $i)
              (i32.const 1)))
          (br 1))))

    (local.set $k
      (i32.const 0))
    (loop
      (if
        (i32.lt_s
          (local.get $k)
          (local.get $K))
        (then
          (local.set $j
            (i32.const 0))
          (loop
            (if
              (i32.lt_s
                (local.get $j)
                (local.get $D))
              (then
                (local.set $frobenius
                  (f64.add
                    (local.get $frobenius)
                    (f64.mul
                      (local.tee $tmp
                        (f64.load
                          (i32.add
                            (local.get $r)
                            (i32.mul
                              (i32.const 8)
                              (i32.add
                                (i32.mul
                                  (local.get $k)
                                  (local.get $D))
                                (local.get $j))))))
                      (local.get $tmp))))
                (local.set $sum_q
                  (f64.add
                    (local.get $sum_q)
                    (f64.load
                      (i32.add
                        (local.get $q)
                        (i32.mul
                          (i32.const 8)
                          (i32.add
                            (i32.mul
                              (local.get $k)
                              (local.get $D))
                            (local.get $j)))))))
                (local.set $j
                  (i32.add
                    (local.get $j)
                    (i32.const 1)))
                (br 1))))
          (local.set $j
            (i32.const 0))
          (loop
            (if
              (i32.lt_s
                (local.get $j)
                (local.get $s))
              (then
                (local.set $frobenius
                  (f64.add
                    (local.get $frobenius)
                    (f64.mul
                      (local.tee $tmp
                        (f64.load
                          (i32.add
                            (local.get $l)
                            (i32.mul
                              (i32.const 8)
                              (i32.add
                                (i32.mul
                                  (local.get $k)
                                  (local.get $s))
                                (local.get $j))))))
                      (local.get $tmp))))
                (local.set $j
                  (i32.add
                    (local.get $j)
                    (i32.const 1)))
                (br 1))))
          (local.set $k
            (i32.add
              (local.get $k)
              (i32.const 1)))
          (br 1))))
    (local.set $n
      (i32.add
        (i32.add
          (local.get $D)
          (local.get $m))
        (i32.const 1)))
    (local.set $log_prior
      (f64.add
        (f64.sub
          (f64.mul
            (f64.convert_i32_s
              (local.get $K))
            (f64.sub
              (f64.mul
                (f64.convert_i32_s
                  (i32.mul
                    (local.get $n)
                    (local.get $D)))
                (call $log
                  (f64.div
                    (local.get $gamma)
                    (f64.sqrt
                      (f64.const 2)))))
              (call $multigammaln
                (local.get $D)
                (f64.div
                  (f64.convert_i32_s
                    (local.get $n))
                  (f64.const 2)))))
          (f64.mul
            (f64.div
              (f64.mul
                (local.get $gamma)
                (local.get $gamma))
              (f64.const 2))
            (local.get $frobenius)))
        (f64.mul
          (f64.convert_i32_s
            (local.get $m))
          (local.get $sum_q))))

    (f64.add
      (local.get $log_likelihood)
      (local.get $log_prior))))

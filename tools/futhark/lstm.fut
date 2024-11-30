def dotproduct [n] (a: [n]f64) (b: [n]f64) : f64 =
  f64.sum <| map2 (*) a b

def sigmoid (x: f64) : f64 =
  1.0f64 / (1.0f64 + f64.exp (-x))

def lstmModel [d]
              (weight: [4][d]f64)
              (bias: [4][d]f64)
              (hidden: [d]f64)
              (cell: [d]f64)
              (input: [d]f64) : ([d]f64, [d]f64) =
  let forget = map sigmoid <| map2 (+) bias[0] <| map2 (*) input weight[0]
  let ingate = map sigmoid <| map2 (+) bias[1] <| map2 (*) hidden weight[1]
  let outgate = map sigmoid <| map2 (+) bias[2] <| map2 (*) input weight[2]
  let change = map f64.tanh <| map2 (+) bias[3] <| map2 (*) hidden weight[3]
  let t1s = map2 (*) cell forget
  let t2s = map2 (*) ingate change
  let cell2 = map2 (+) t1s t2s
  let hidden2 = map2 (*) outgate <| map f64.tanh cell2
  in (hidden2, cell2)

def lstmPredict [slen] [d]
                (mainParams: [slen][2][4][d]f64)
                (extraParams: [3][d]f64)
                (state: [slen][2][d]f64)
                (input: [d]f64) : ([d]f64, [slen][2][d]f64) =
  let x0 = map2 (*) input extraParams[0]
  let state_ini = replicate slen <| replicate 2 <| replicate d 0f64
  -- : [slen][2][d]f64
  let (state', x') =
    loop (s, x) = (state_ini, x0)
    for i < slen do
      let (h, c) = lstmModel mainParams[i, 0] mainParams[i, 1] state[i, 0] state[i, 1] x
      let s[i, 0] = h
      let s[i, 1] = c
      in (s, h)
  let v' =
    map2 (*) x' extraParams[1]
    |> map2 (+) extraParams[2]
  in (v', state')

def lstmObjective [stlen] [lenSeq] [d]
                  (mainParams0: [stlen * 2][4 * d]f64)
                  (extraParams: [3][d]f64)
                  (state0: [stlen * 2][d]f64)
                  (sequence: [lenSeq][d]f64) : f64 =
  -- mainParams : [stlen][2][4][d]f64
  let mainParams = unflatten <| map unflatten mainParams0
  -- state : [stlen][2][d]f64
  let state = unflatten state0
  let (_, total) =
    loop (oldState, oldTotal) = (state, 0f64)
    for i < lenSeq - 1 do
      let (y_pred, newState) = lstmPredict mainParams extraParams oldState sequence[i]
      -- y_pred: DV [d]f64, newState: DM
      let tmp_sum = f64.sum <| map f64.exp y_pred
      let tmp_log = -f64.log (tmp_sum + 2.0f64)
      let ynorm = map (+ tmp_log) y_pred
      let newTotal = oldTotal + (dotproduct sequence[i + 1] ynorm)
      in (newState, newTotal)
  let count = d * (lenSeq - 1)
  in -(total / f64.i64 (count))

entry calculate_objective [stlenx2] [lenSeq] [d]
                          (mainParams0: [stlenx2][4 * d]f64)
                          (extraParams: [3][d]f64)
                          (state0: [stlenx2][d]f64)
                          (sequence: [lenSeq][d]f64) : f64 =
  let stlen = stlenx2 / 2
  in lstmObjective (sized (stlen * 2) mainParams0)
                   extraParams
                   (sized (stlen * 2) state0)
                   sequence

entry calculate_jacobian [stlenx2] [lenSeq] [d]
                         (mainParams: [stlenx2][4 * d]f64)
                         (extraParams: [3][d]f64)
                         (state0: [stlenx2][d]f64)
                         (sequence: [lenSeq][d]f64) : []f64 =
  let (x, y) =
    vjp (\(x, y) -> calculate_objective x y state0 sequence) (mainParams, extraParams) 1
  in flatten x ++ flatten y

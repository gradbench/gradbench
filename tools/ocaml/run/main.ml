open Gradbench_evals

type id = int
type eval_name = string
type function_name = string
type module_name = string

type msg = Msg_Start of id * eval_name
         | Msg_Define of id * module_name
         | Msg_Evaluate of id * module_name * function_name * Yojson.Basic.t
         | Msg_Unknown of id

let msg_id = function
  | Msg_Start (id, _) -> id
  | Msg_Define (id, _) -> id
  | Msg_Evaluate (id, _, _, _) -> id
  | Msg_Unknown id -> id

let look k l = snd (List.find (fun x -> fst x = k) l)

let look_string k l =
  match look k l with
    (`String x) -> x
  | _ -> failwith "look_string: not a string"

let look_int k l =
  match look k l with
    (`Int x) -> x
  | _ -> failwith "look_int: not an int"

let look_float k l =
  match look k l with
    (`Int x) -> float_of_int x
  | (`Float x) -> x
  | _ -> failwith "look_float: not a number"

let msg_of_json = function
  | `Assoc l ->
     let id = look_int "id" l in
     let kind = look_string "kind" l in
     (match kind with
       "start" ->
        let eval = look_string "eval" l in
        Msg_Start (id, eval)
     | "define" ->
        let m = look_string "module" l in
        Msg_Define (id, m)
     | "evaluate" ->
        let m = look_string "module" l in
        let f = look_string "function" l in
        let input = look "input" l in
        Msg_Evaluate (id, m, f, input)
     | _ -> Msg_Unknown id)
  | _ -> failwith "msg_of_json: not an object"

type runs = {minRuns: int; minSeconds: float}

let runs_of = function
    `Assoc l -> {minRuns = look_int "min_runs" l;
                 minSeconds = look_float "min_seconds" l}
  | _ -> {minRuns = 1; minSeconds = 0.0}

let nanoseconds() = int_of_float(Unix.gettimeofday() *. 1.0e9)

let rec do_runs timings i elapsed runs f x =
  let bef = nanoseconds() in
  let output = f x in
  let aft = nanoseconds() in
  let ns = aft-bef in
  let timings' = ns :: timings in
  let elapsed' = elapsed +. float_of_int ns/.1e9
  in if i < runs.minRuns || elapsed' < runs.minSeconds
     then do_runs timings' (i+1) elapsed' runs f x
     else (output, timings')

let wrap f input_from_json output_to_json input =
  let input' = input_from_json input in
  let (output, timings) = do_runs [] 1 0. (runs_of input) f input'
  in (output_to_json output, timings)

let modules =
  [ (let module M = Evals_effect_handlers_hello.Hello
     in (("hello", "square"), wrap M.square M.input_of_json M.json_of_output));
    (let module M = Evals_effect_handlers_hello.Hello
    in (("hello", "double"), wrap M.double M.input_of_json M.json_of_output));
    (let module M = Evals_effect_handlers_gmm.GMM
     in (("gmm", "objective"), wrap M.objective M.input_of_json M.json_of_objective));
    (let module M = Evals_effect_handlers_gmm.GMM
     in (("gmm", "jacobian"), wrap M.jacobian M.input_of_json M.json_of_jacobian));
    (let module M = Evals_effect_handlers_lse.LSE
     in (("lse", "primal"), wrap M.primal M.input_of_json M.json_of_primal));
    (let module M = Evals_effect_handlers_lse.LSE
     in (("lse", "gradient"), wrap M.gradient M.input_of_json M.json_of_gradient));
  ]

let timing_of x =
  `Assoc [("name", `String "evaluate"); ("nanoseconds", `Int x)]

let () =
  while true; do
    match In_channel.input_line stdin with
    | Some line ->
       let msg = msg_of_json (Yojson.Basic.from_string line) in
       let reply (vs: (string * Yojson.Basic.t) list) =
         (Yojson.Basic.to_channel stdout (`Assoc (("id", `Int (msg_id msg)) :: vs));
          Printf.printf "\n%!")
       in (match msg with
            Msg_Unknown _id -> reply []
          | Msg_Start (_id, _eval) -> reply [("tool", `String "ocaml")]
          | Msg_Define (_id, mname) ->
             reply [("success", `Bool (List.exists (fun l -> fst (fst l) = mname) modules))]
          | Msg_Evaluate (_id, mname, fname, input) ->
             match look (mname, fname) modules with
               f ->
               let (output, timings) = f input in
               let timings' = `List (List.map timing_of timings)
               in reply [("success", `Bool true);
                         ("output",output);
                         ("timings", timings')])
    | None -> exit 0
  done

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

let () =
  while true; do
    match In_channel.input_line stdin with
    | Some line ->
       let msg = msg_of_json (Yojson.Basic.from_string line) in
       let reply vs =
         (Yojson.Basic.to_channel stdout (`Assoc (("id", `Int (msg_id msg)) :: vs));
          Printf.printf "\n%!")
       in (match msg with
             Msg_Unknown _id -> reply []
           | Msg_Start (_id, _eval) -> reply [("tool", `String "ocaml")]
           | Msg_Define (_id, _f) -> reply [("success", `Bool false)]
           | Msg_Evaluate (_id, _mod, _, _input) -> reply [("success", `Bool false)])
    | None -> exit 0
  done

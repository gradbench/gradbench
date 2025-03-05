type id = int

type msg = Msg_Start of id * string
         | Msg_Unknown of id

let msg_of_json = function
  | `Assoc l ->
     let (_, (`Int id : Yojson.Basic.t)) = List.find (fun x -> fst x = "id") l in
     let (_, (`String kind : Yojson.Basic.t)) = List.find (fun x -> fst x = "kind") l in
     (match kind with
        "start" ->
         let (_, (`String eval : Yojson.Basic.t)) = List.find (fun x -> fst x = "eval") l in
         Msg_Start (id, eval)
      | _ -> Msg_Unknown id)
  | _ -> failwith "msg_of_json: invalid"

let () =
  while true; do
    match In_channel.input_line stdin with
    | Some line ->
       let json = Yojson.Basic.from_string line in
       let msg = msg_of_json json
       in ()
    | None -> exit 0
  done

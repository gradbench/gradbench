(* https://github.com/jasigal/ADBench/blob/b98752f96a3b785e07ff6991853dc1073e6bf075/src/ocaml/shared/shared_gmm_types.ml *)

(* Wishart priors *)
type 's wishart = {
  gamma : 's ;
  m : int;
}

(* GMM data as arrays *)
type ('s, 't) gmm_input = {
  alphas : 't;
  mu : 't;
  q: 't;
  l: 't;
  x : 't;
  wishart : 's wishart;
}

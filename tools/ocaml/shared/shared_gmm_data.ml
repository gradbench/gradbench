(* Wishart priors *)
type 's wishart = {
  gamma : 's ;
  m : int;
}

(* GMM data as arrays *)
type ('s, 't) gmm_input = {
  alphas : 't;
  means : 't;
  icfs : 't;
  x : 't;
  wishart : 's wishart;
}

(* Output data *)
type ('s, 't) gmm_output = {
  objective : 's;
  gradient : 't;
}

(* Parameters *)
type gmm_parameters = {
  replicate_point : bool;
}
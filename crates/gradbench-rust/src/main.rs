mod matrix;

use std::{
    collections::HashMap,
    f64::consts::PI,
    io,
    time::{Duration, Instant},
};

use matrix::Matrix;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

type Id = i64;

#[derive(Deserialize)]
struct Message {
    id: Id,
    kind: String,
    module: Option<String>,
}

#[derive(Serialize)]
struct Response {
    id: Id,
}

#[derive(Serialize)]
struct Timing {
    name: &'static str,
    nanoseconds: u128,
}

#[derive(Serialize)]
struct DefineResponse {
    id: Id,
    success: bool,
}

#[derive(Serialize)]
struct EvaluateResponse<'a, T> {
    id: Id,
    success: bool,
    output: T,
    timings: &'a [Timing],
}

fn print_jsonl(value: &impl Serialize) {
    serde_json::to_writer(&mut io::stdout(), value).unwrap();
    println!();
}

fn start(line: &str) {
    let Message { kind, id, .. } = serde_json::from_str(line).unwrap();
    assert_eq!(kind, "start");
    print_jsonl(&Response { id });
}

fn respond<T: Serialize>(id: Id, output: T, durations: &[Duration]) {
    let timings: Vec<Timing> = durations
        .iter()
        .map(|duration| Timing {
            name: "evaluate",
            nanoseconds: duration.as_nanos(),
        })
        .collect();
    print_jsonl(&EvaluateResponse {
        id,
        success: true,
        output,
        timings: &timings,
    });
}

struct Context {}

trait GradBenchModule {
    fn evaluate(&self, context: &mut Context, id: Id, line: &str);
}

fn main() {
    let mut context = Context {};
    let mut modules = HashMap::<String, Box<dyn GradBenchModule>>::new();
    let mut lines = io::stdin().lines();
    start(&lines.next().unwrap().unwrap());
    for result in lines {
        let line = result.unwrap();
        let message: Message = serde_json::from_str(&line).unwrap();
        let id = message.id;
        match message.kind.as_str() {
            "define" => {
                let module = message.module.unwrap();
                let mut defining = Defining::new();
                let success = defining
                    .module(&module)
                    .map(|defined| modules.insert(module, defined))
                    .is_some();
                print_jsonl(&DefineResponse { id, success });
            }
            "evaluate" => modules[&message.module.unwrap()].evaluate(&mut context, id, &line),
            _ => print_jsonl(&Response { id }),
        }
    }
}

struct Defining {}

impl Defining {
    fn new() -> Self {
        Self {}
    }

    fn module(&mut self, name: &str) -> Option<Box<dyn GradBenchModule>> {
        match name {
            "hello" => Some(Box::new(Hello {})),
            "gmm" => Some(Box::new(Gmm {})),
            _ => None,
        }
    }
}

struct Hello {}

#[derive(Deserialize)]
#[serde(tag = "function", rename_all = "snake_case")]
enum HelloMessage {
    Square { input: f64 },
    Double { input: f64 },
}

impl GradBenchModule for Hello {
    fn evaluate(&self, _: &mut Context, id: Id, line: &str) {
        match serde_json::from_str::<HelloMessage>(line).unwrap() {
            HelloMessage::Square { input } => {
                let start = Instant::now();
                let output = input * input;
                respond(id, output, &[start.elapsed()]);
            }
            HelloMessage::Double { input } => {
                let start = Instant::now();
                let output = input + input;
                respond(id, output, &[start.elapsed()]);
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct GmmInput {
    min_runs: usize,
    min_seconds: u64,
    d: usize,
    k: usize,
    n: usize,
    x: Matrix,
    m: usize,
    gamma: f64,
    alpha: Vec<f64>,
    mu: Matrix,
    q: Matrix,
    l: Matrix,
}

struct Gmm {}

#[derive(Deserialize)]
#[serde(tag = "function", rename_all = "snake_case")]
enum GmmMessage {
    Objective { input: GmmInput },
    Jacobian { input: GmmInput },
}

impl GradBenchModule for Gmm {
    fn evaluate(&self, _: &mut Context, id: Id, line: &str) {
        match serde_json::from_str::<GmmMessage>(line).unwrap() {
            GmmMessage::Objective {
                input:
                    GmmInput {
                        min_runs,
                        min_seconds,
                        d,
                        k,
                        n,
                        x,
                        m,
                        gamma,
                        alpha,
                        mu,
                        q,
                        l,
                    },
            } => {
                let mut l_by_rows = Matrix::new(l.rows(), l.cols());
                for kk in 0..k {
                    let mut i = 0;
                    for c in 0..d {
                        for r in (c + 1)..d {
                            let j = (r * (r - 1)) / 2;
                            l_by_rows[(kk, j + c)] = l[(kk, i)];
                            i += 1;
                        }
                    }
                }
                let mut total = Duration::ZERO;
                let mut durations = Vec::new();
                loop {
                    let start = Instant::now();
                    let output = gmm(d, k, n, &x, m, gamma, &alpha, &mu, &q, &l_by_rows);
                    let duration = start.elapsed();
                    total += duration;
                    durations.push(duration);
                    if durations.len() >= min_runs && total >= Duration::from_secs(min_seconds) {
                        respond(id, output, &durations);
                        return;
                    }
                }
            }
            GmmMessage::Jacobian { .. } => {
                let start = Instant::now();
                let output = ();
                respond(id, output, &[start.elapsed()]);
            }
        }
    }
}

fn multigammaln(z: f64, p: usize) -> f64 {
    assert!(p >= 1);
    debug_assert!(z > (p as f64 - 1.0) * 0.5);
    let coeff = 0.25 * (p * (p - 1)) as f64 * PI.ln();
    let sum = (1..=p)
        .map(|j| ln_gamma(z + 0.5 * (1. - j as f64)))
        .sum::<f64>();
    coeff + sum
}

fn maximum(x: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &xi in x {
        if xi > max {
            max = xi;
        }
    }
    max
}

fn logsumexp(x: &[f64]) -> f64 {
    let mut sum = 0.;
    let a = maximum(x);
    for &xi in x {
        sum += (xi - a).exp();
    }
    a + sum.ln()
}

fn quadratic_form_element(mu: &[f64], r: &[f64], l: &[f64], x: &[f64], i: usize) -> f64 {
    let k = (i * (i - 1)) / 2;
    let mut e = 0.;
    for j in 0..i {
        e += l[k + j] * (x[j] - mu[j]);
    }
    e + r[i] * (x[i] - mu[i])
}

fn gmm(
    D: usize,
    K: usize,
    N: usize,
    x: &Matrix,
    m: usize,
    gamma: f64,
    alpha: &[f64],
    mu: &Matrix,
    q: &Matrix,
    l: &Matrix,
) -> f64 {
    let mut r = Matrix::new(K, D);
    for k in 0..K {
        for j in 0..D {
            r[(k, j)] = q[(k, j)].exp();
        }
    }

    let mut log_likelihood =
        -(N as f64) * (((D as f64) / 2.0) * (2.0 * PI).ln() + logsumexp(alpha));
    for i in 0..N {
        let x_i = x.row(i);
        let mut beta = vec![0.; K];
        for k in 0..K {
            let mu_k = mu.row(k);
            let r_k = r.row(k);
            let l_k = l.row(k);
            let mut normsq = 0.;
            let mut sum = 0.;
            for j in 0..D {
                let e = quadratic_form_element(mu_k, r_k, l_k, x_i, j);
                normsq += e * e;
                sum += q[(k, j)];
            }
            beta[k] = alpha[k] - 0.5 * normsq + sum;
        }
        log_likelihood += logsumexp(&beta);
    }

    let mut frobenius = 0.;
    let mut sum_q = 0.;
    for k in 0..K {
        for j in 0..D {
            let r_kj = r[(k, j)];
            frobenius += r_kj * r_kj;
            let q_kj = q[(k, j)];
            sum_q += q_kj * q_kj;
        }
        let s = l.cols();
        for j in 0..s {
            let l_kj = l[(k, j)];
            frobenius += l_kj * l_kj;
        }
    }
    let n = D + m + 1;
    let log_prior = (K as f64)
        * (((n * D) as f64) * (gamma / (2f64).sqrt()).ln() - multigammaln((n as f64) / 2., D))
        - (gamma * gamma / 2.) * frobenius
        + (m as f64) * sum_q;

    log_likelihood + log_prior
}

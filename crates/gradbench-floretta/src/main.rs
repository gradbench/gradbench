mod hello;
mod kmeans;

use std::{collections::HashMap, io, time::Instant};

use serde::{Deserialize, Serialize};
use wasmtime::{Engine, Store};

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
    timings: Vec<Timing>,
}

#[derive(Serialize)]
struct EvaluateResponse<T> {
    id: Id,
    success: bool,
    output: T,
    timings: Vec<Timing>,
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

fn respond<T: Serialize>(id: Id, output: T, timings: Vec<Timing>) {
    print_jsonl(&EvaluateResponse {
        id,
        success: true,
        output,
        timings,
    });
}

struct Context {
    engine: Engine,
    store: Store<()>,
}

trait GradBenchModule {
    fn evaluate(&self, context: &mut Context, id: Id, line: &str);
}

fn main() {
    let engine = Engine::default();
    let store = Store::new(&engine, ());
    let mut context = Context { engine, store };
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
                let mut defining = Defining::new(&mut context);
                let success = defining
                    .module(&module)
                    .map(|defined| modules.insert(module, defined))
                    .is_some();
                print_jsonl(&DefineResponse {
                    id,
                    success,
                    timings: defining.timings,
                });
            }
            "evaluate" => modules[&message.module.unwrap()].evaluate(&mut context, id, &line),
            _ => print_jsonl(&Response { id }),
        }
    }
}

struct Defining<'a> {
    context: &'a mut Context,
    timings: Vec<Timing>,
}

impl<'a> Defining<'a> {
    fn new(context: &'a mut Context) -> Self {
        Self {
            context,
            timings: Vec::new(),
        }
    }

    fn time<T>(&mut self, name: &'static str, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        self.timings.push(Timing {
            name,
            nanoseconds: start.elapsed().as_nanos(),
        });
        result
    }

    fn module(&mut self, name: &str) -> Option<Box<dyn GradBenchModule>> {
        match name {
            "hello" => Some(self.hello()),
            "kmeans" => Some(self.kmeans()),
            _ => None,
        }
    }
}

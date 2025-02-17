use std::{collections::HashMap, io, time::Instant};

use serde::{Deserialize, Serialize};
use wasmtime::{Engine, Instance, Module, Store, TypedFunc};

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

fn respond<T: Serialize>(id: Id, output: T) {
    print_jsonl(&EvaluateResponse {
        id,
        success: true,
        output,
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
            "hello" => {
                let wasm_original = wat::parse_file("tools/floretta/hello.wat").unwrap();
                let wasm = self.time("autodiff", || {
                    let mut ad = floretta::Autodiff::new();
                    ad.export("square", "backprop");
                    ad.transform(&wasm_original).unwrap()
                });
                let module = Module::new(&self.context.engine, &wasm).unwrap();
                let instance = Instance::new(&mut self.context.store, &module, &[]).unwrap();
                let square = instance
                    .get_typed_func::<f64, f64>(&mut self.context.store, "square")
                    .unwrap();
                let backprop = instance
                    .get_typed_func::<f64, f64>(&mut self.context.store, "backprop")
                    .unwrap();
                Some(Box::new(Hello { square, backprop }))
            }
            _ => None,
        }
    }
}

struct Hello {
    square: TypedFunc<f64, f64>,
    backprop: TypedFunc<f64, f64>,
}

#[derive(Deserialize)]
#[serde(tag = "function", rename_all = "snake_case")]
enum HelloMessage {
    Square { input: f64 },
    Double { input: f64 },
}

impl GradBenchModule for Hello {
    fn evaluate(&self, context: &mut Context, id: Id, line: &str) {
        match serde_json::from_str::<HelloMessage>(line).unwrap() {
            HelloMessage::Square { input } => {
                let output = self.square.call(&mut context.store, input).unwrap();
                respond(id, output);
            }
            HelloMessage::Double { input } => {
                self.square.call(&mut context.store, input).unwrap();
                let output = self.backprop.call(&mut context.store, 1.).unwrap();
                respond(id, output);
            }
        }
    }
}

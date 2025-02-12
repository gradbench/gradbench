use std::{collections::HashMap, io};

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
struct DefineResponse {
    id: Id,
    success: bool,
}

#[derive(Serialize)]
struct EvaluateResponse<T> {
    id: Id,
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
                let success = define(&mut context, &module)
                    .map(|defined| modules.insert(module, defined))
                    .is_some();
                print_jsonl(&DefineResponse { id, success });
            }
            "evaluate" => modules[&message.module.unwrap()].evaluate(&mut context, id, &line),
            _ => print_jsonl(&Response { id }),
        }
    }
}

fn define(context: &mut Context, name: &str) -> Option<Box<dyn GradBenchModule>> {
    match name {
        "hello" => {
            let mut ad = floretta::Autodiff::new();
            ad.export("square", "backprop");
            let wasm = ad
                .transform(&wat::parse_file("tools/floretta/hello.wat").unwrap())
                .unwrap();
            let module = Module::new(&context.engine, &wasm).unwrap();
            let instance = Instance::new(&mut context.store, &module, &[]).unwrap();
            let square = instance
                .get_typed_func::<f64, f64>(&mut context.store, "square")
                .unwrap();
            let backprop = instance
                .get_typed_func::<f64, f64>(&mut context.store, "backprop")
                .unwrap();
            Some(Box::new(Hello { square, backprop }))
        }
        _ => None,
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
                print_jsonl(&EvaluateResponse { id, output });
            }
            HelloMessage::Double { input } => {
                self.square.call(&mut context.store, input).unwrap();
                let output = self.backprop.call(&mut context.store, 1.).unwrap();
                print_jsonl(&EvaluateResponse { id, output });
            }
        }
    }
}

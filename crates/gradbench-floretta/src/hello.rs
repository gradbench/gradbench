use serde::Deserialize;
use wasmtime::{Instance, Module, TypedFunc};

use crate::{respond, Context, Defining, GradBenchModule, Id};

impl Defining<'_> {
    pub fn hello(&mut self) -> Box<dyn GradBenchModule> {
        let wasm_original = wat::parse_file("tools/floretta/hello.wat").unwrap();
        let wasm = self.time("autodiff", || {
            let mut ad = floretta::Autodiff::new();
            ad.export("square", "backprop");
            ad.transform(&wasm_original).unwrap()
        });
        let module = Module::new(&self.context.engine, &wasm).unwrap();
        let instance = Instance::new(&mut self.context.store, &module, &[]).unwrap();
        let square = instance
            .get_typed_func(&mut self.context.store, "square")
            .unwrap();
        let backprop = instance
            .get_typed_func(&mut self.context.store, "backprop")
            .unwrap();
        Box::new(Hello { square, backprop })
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

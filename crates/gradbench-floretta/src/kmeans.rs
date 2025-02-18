use serde::Deserialize;
use wasmtime::{Instance, Memory, Module, TypedFunc};

use crate::{respond, Context, Defining, GradBenchModule, Id};

impl Defining<'_> {
    pub fn kmeans(&mut self) -> Box<dyn GradBenchModule> {
        let wasm = wat::parse_file("tools/floretta/kmeans.wat").unwrap();
        let module = Module::new(&self.context.engine, &wasm).unwrap();
        let instance = Instance::new(&mut self.context.store, &module, &[]).unwrap();
        let memory = instance
            .get_memory(&mut self.context.store, "memory")
            .unwrap();
        let cost = instance
            .get_typed_func(&mut self.context.store, "cost")
            .unwrap();
        Box::new(KMeans { memory, cost })
    }
}

type Params = (i32, i32, i32, i32, i32);

struct KMeans {
    memory: Memory,
    cost: TypedFunc<Params, f64>,
}

type Matrix = Vec<Vec<f64>>;

fn shape(matrix: &Matrix) -> (usize, usize) {
    (matrix.len(), matrix[0].len())
}

#[derive(Deserialize)]
struct KMeansInput {
    points: Matrix,
    centroids: Matrix,
}

#[derive(Deserialize)]
#[serde(tag = "function", rename_all = "snake_case")]
enum KMeansMessage {
    Cost { input: KMeansInput },
    Dir { input: KMeansInput },
}

impl KMeans {
    fn accommodate(&self, context: &mut Context, bytes: usize) {
        let pages = u64::try_from(bytes)
            .unwrap()
            .div_ceil(self.memory.page_size(&context.store));
        let current = self.memory.size(&context.store);
        self.memory
            .grow(&mut context.store, pages.saturating_sub(current))
            .unwrap();
    }

    fn store(&self, context: &mut Context, offset: &mut usize, matrix: Matrix) -> usize {
        let pointer = *offset;
        let (rows, cols) = shape(&matrix);
        self.accommodate(context, rows * cols * size_of::<f64>());
        for row in matrix {
            for value in row {
                self.memory
                    .write(&mut context.store, *offset, &value.to_le_bytes())
                    .unwrap();
                *offset += size_of::<f64>();
            }
        }
        pointer
    }

    fn prepare(&self, context: &mut Context, input: KMeansInput) -> Params {
        let (k, d) = shape(&input.centroids);
        let (n, d1) = shape(&input.points);
        assert_eq!(d, d1);
        let mut offset = 0;
        let c = self.store(context, &mut offset, input.centroids);
        let p = self.store(context, &mut offset, input.points);
        (
            i32::try_from(d).unwrap(),
            i32::try_from(k).unwrap(),
            i32::try_from(n).unwrap(),
            i32::try_from(c).unwrap(),
            i32::try_from(p).unwrap(),
        )
    }
}

impl GradBenchModule for KMeans {
    fn evaluate(&self, context: &mut Context, id: Id, line: &str) {
        match serde_json::from_str::<KMeansMessage>(line).unwrap() {
            KMeansMessage::Cost { input } => {
                let params = self.prepare(context, input);
                let output = self.cost.call(&mut context.store, params).unwrap();
                respond(id, output);
            }
            KMeansMessage::Dir { input } => {
                self.prepare(context, input);
                todo!()
            }
        }
    }
}

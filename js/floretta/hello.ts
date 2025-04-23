import { sh } from "./util.ts";

const wasm = await sh("wasm-tools parse tools/floretta/hello.wat");
const module = await WebAssembly.instantiate(wasm);
export const square = (x: number) => {
  const { square } = module.instance.exports as {
    square: (x: number) => number;
  };
  return { output: square(x) };
};

const wasmGrad = await sh(
  "floretta --reverse tools/floretta/hello.wat --export square backprop",
);
const moduleGrad = await WebAssembly.instantiate(wasmGrad);
export const double = (x: number) => {
  const { square: forward, backprop } = moduleGrad.instance.exports as {
    square: (x: number) => number;
    backprop: (dy: number) => number;
  };
  forward(x);
  return { output: backprop(1) };
};

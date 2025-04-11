import { sh } from "./util.js";

const wasm = await sh("wasm-tools parse tools/floretta/hello.wat");
const module = await WebAssembly.instantiate(wasm);
export const square = (x) => {
  const { square } = module.instance.exports;
  return { output: square(x) };
};

const wasmGrad = await sh(
  "floretta --reverse tools/floretta/hello.wat --export square backprop",
);
const moduleGrad = await WebAssembly.instantiate(wasmGrad);
export const double = (x) => {
  const { square: forward, backprop } = moduleGrad.instance.exports;
  forward(x);
  return { output: backprop(1) };
};

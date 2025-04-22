import { accommodate, multipleRuns, sh } from "./util.js";

const wasm = await sh("wasm-tools parse tools/floretta/llsq.wat");
const module = await WebAssembly.instantiate(wasm);
export const primal = multipleRuns(({ x, n }) => {
  const m = x.length;
  const { memory, llsq } = module.instance.exports;
  const bytes = Float64Array.BYTES_PER_ELEMENT * m;
  accommodate(memory, bytes);
  return () => {
    new Float64Array(memory.buffer).set(x);
    return llsq(n, m, 0);
  };
});

const wasmGrad = await sh(
  "floretta --reverse tools/floretta/llsq.wat --export memory memory_bwd --export llsq backprop",
);
const moduleGrad = await WebAssembly.instantiate(wasmGrad);
export const gradient = multipleRuns(({ x, n }) => {
  const m = x.length;
  const { memory, memory_bwd, llsq, backprop } = moduleGrad.instance.exports;
  const bytes = Float64Array.BYTES_PER_ELEMENT * m;
  accommodate(memory, bytes);
  accommodate(memory_bwd, bytes);
  return () => {
    new Float64Array(memory.buffer).set(x);
    llsq(n, m, 0);
    const grad = new Float64Array(memory_bwd.buffer, 0, m);
    grad.fill(0);
    backprop(1);
    return Array.from(grad);
  };
});

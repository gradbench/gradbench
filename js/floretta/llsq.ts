import type { Runs } from "./protocol.ts";
import { accommodate, getExports, multipleRuns, sh } from "./util.ts";

interface Input extends Runs {
  x: number[];
  n: number;
}

interface Exports {
  memory: WebAssembly.Memory;
  llsq: (n: number, m: number, x: number) => number;
}

interface ExportsGrad extends Exports {
  memory_bwd: WebAssembly.Memory;
  backprop: (dy: number) => void;
}

const wasm = await sh("wasm-tools parse tools/floretta/llsq.wat");
const module = await WebAssembly.instantiate(wasm);
export const primal = multipleRuns(({ x, n }: Input): (() => number) => {
  const m = x.length;
  const { memory, llsq } = getExports<Exports>(module);
  const bytes = Float64Array.BYTES_PER_ELEMENT * m;
  accommodate(memory, bytes);
  return () => {
    new Float64Array(memory.buffer).set(x);
    return llsq(n, m, 0);
  };
});

const wasmGrad = await sh("wasm-tools parse tools/floretta/llsq_grad.wat");
const moduleGrad = await WebAssembly.instantiate(wasmGrad);
export const gradient = multipleRuns(({ x, n }: Input): (() => number[]) => {
  const m = x.length;
  const { memory, memory_bwd, llsq, backprop } =
    getExports<ExportsGrad>(moduleGrad);
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

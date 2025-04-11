import { accommodate, multipleRuns, sh } from "./util.js";

const wasm = await sh("wasm-tools parse tools/floretta/lse.wat");
const module = await WebAssembly.instantiate(wasm, {
  math: { exp: Math.exp, log: Math.log },
});
export const primal = multipleRuns(({ x }) => {
  const { memory, logsumexp } = module.instance.exports;
  const bytes = Float64Array.BYTES_PER_ELEMENT * x.length;
  accommodate(memory, bytes);
  return () => {
    new Float64Array(memory.buffer).set(x);
    return logsumexp(x.length, 0);
  };
});

// const wasmGrad = await sh("wasm-tools parse tools/floretta/lse_grad.wat");
// const moduleGrad = await WebAssembly.instantiate(wasmGrad, {
//   math: { exp: Math.exp, log: Math.log },
// });
const wasmGrad = await sh(
  "floretta --reverse tools/floretta/lse.wat --import math exp math exp_bwd --import math log math log_bwd --export memory memory_bwd --export logsumexp backprop",
);
const tape = [];
const moduleGrad = await WebAssembly.instantiate(wasmGrad, {
  math: {
    exp: (x) => {
      const y = Math.exp(x);
      tape.push(y);
      return y;
    },
    exp_bwd: (dy) => {
      const y = tape.pop();
      return dy * y;
    },
    log: (x) => {
      const y = Math.log(x);
      tape.push(y);
      return y;
    },
    log_bwd: (dy) => {
      const y = tape.pop();
      return dy / y;
    },
  },
});
export const gradient = multipleRuns(({ x }) => {
  const { memory, memory_bwd, logsumexp, backprop } =
    moduleGrad.instance.exports;
  const bytes = Float64Array.BYTES_PER_ELEMENT * x.length;
  accommodate(memory, bytes);
  accommodate(memory_bwd, bytes);
  return () => {
    new Float64Array(memory.buffer).set(x);
    logsumexp(x.length, 0);
    const grad = new Float64Array(memory_bwd.buffer, 0, x.length);
    grad.fill(0);
    backprop(1);
    return Array.from(grad);
  };
});

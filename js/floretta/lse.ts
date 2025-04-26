import type { Runs } from "./protocol.ts";
import { accommodate, getExports, multipleRuns, sh } from "./util.ts";

interface Input extends Runs {
  x: number[];
}

interface Exports {
  memory: WebAssembly.Memory;
  logsumexp: (n: number, x: number) => number;
}

interface ExportsGrad extends Exports {
  memory_bwd: WebAssembly.Memory;
  backprop: (dlse: number) => void;
}

const wasm = await sh("wasm-tools parse tools/floretta/lse.wat");
const module = await WebAssembly.instantiate(wasm, {
  math: { exp: Math.exp, log: Math.log },
});
export const primal = multipleRuns(({ x }: Input): (() => number) => {
  const { memory, logsumexp } = getExports<Exports>(module);
  const bytes = Float64Array.BYTES_PER_ELEMENT * x.length;
  accommodate(memory, bytes);
  return () => {
    new Float64Array(memory.buffer).set(x);
    return logsumexp(x.length, 0);
  };
});

const wasmGrad = await sh(
  "floretta --reverse tools/floretta/lse.wat --import math exp math exp_bwd --import math log math log_bwd --export memory memory_bwd --export logsumexp backprop",
);
const tape: number[] = [];
const moduleGrad = await WebAssembly.instantiate(wasmGrad, {
  math: {
    exp: (x: number) => {
      const y = Math.exp(x);
      tape.push(y);
      return y;
    },
    exp_bwd: (dy: number) => {
      const y = tape.pop()!;
      return dy * y;
    },
    log: (x: number) => {
      tape.push(x);
      return Math.log(x);
    },
    log_bwd: (dy: number) => {
      const x = tape.pop()!;
      return dy / x;
    },
  },
});
export const gradient = multipleRuns(({ x }: Input): (() => number[]) => {
  const { memory, memory_bwd, logsumexp, backprop } =
    getExports<ExportsGrad>(moduleGrad);
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

import type { Independent, Input } from "@gradbench/common/gmm.ts";
import type { Float, Int } from "@gradbench/common/protocol.ts";
import { multipleRuns } from "@gradbench/common/util.ts";
import { getExports, sh } from "./util.ts";

interface Exports {
  memory: WebAssembly.Memory;
  free8: WebAssembly.Global;
  malloc: (words: Int) => Int;
  gmm: (
    D: Int,
    K: Int,
    N: Int,
    x: Int,
    m: Int,
    gamma: Float,
    alpha: Int,
    mu: Int,
    q: Int,
    l: Int,
  ) => Float;
}

const wasm = await sh("wasm-tools parse tools/floretta/gmm.wat");
const module = await WebAssembly.instantiate(wasm, {
  math: {
    exp: Math.exp,
    log: Math.log,
    multigammaln: (p: Int, a: Float): Float => 1.5963125911388552, // TODO
  },
});
export const objective = multipleRuns(
  ({ d: D, k: K, n: N, x, m, gamma, alpha, mu, q, l }: Input) => {
    const s = (D * (D - 1)) / 2;
    const { memory, free8, malloc, gmm } = getExports<Exports>(module);

    const ptrX = malloc(N * D);
    const ptrAlpha = malloc(K);
    const ptrMu = malloc(K * D);
    const ptrQ = malloc(K * D);
    const ptrL = malloc(K * s);
    const ptr = free8.value;

    const words = new Float64Array(memory.buffer);
    const memcpy = (src: Float[], dst: Int, i: Int = 0) =>
      words.set(src, dst / 8 + i);
    for (let i = 0; i < N; ++i) memcpy(x[i], ptrX, i * D);
    memcpy(alpha, ptrAlpha);
    for (let k = 0; k < K; ++k) memcpy(mu[k], ptrMu, k * D);
    for (let k = 0; k < K; ++k) memcpy(q[k], ptrQ, k * D);
    for (let k = 0; k < K; ++k) memcpy(l[k], ptrL, k * s); // TODO

    return (): Float => {
      free8.value = ptr;
      return gmm(D, K, N, ptrX, m, gamma, ptrAlpha, ptrMu, ptrQ, ptrL);
    };
  },
);

export const jacobian = multipleRuns(({}: Input) => {
  return (): Independent => {
    throw Error("TODO");
  };
});

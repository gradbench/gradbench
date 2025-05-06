import type { Runs } from "@gradbench/common/protocol.ts";
import { multipleRuns } from "@gradbench/common/util.ts";
import * as tf from "@tensorflow/tfjs";

const f =
  (n: number) =>
  (x: tf.Tensor): tf.Tensor => {
    let t = tf
      .range(0, n)
      .mul(2 / (n - 1))
      .sub(1);
    let s = t.sign();
    let c = tf.onesLike(t);
    for (let j = 0; j < x.size; ++j) {
      const xj = x.slice(j, 1);
      s = s.sub(c.mul(xj));
      c = c.mul(t);
    }
    return s.square().sum().div(2);
  };

interface Input extends Runs {
  x: number[];
  n: number;
}

export const primal = multipleRuns(({ x, n }: Input) => {
  return () => f(n)(tf.tensor(x)).arraySync();
});

export const gradient = multipleRuns(({ x, n }: Input) => {
  const g = tf.grad(f(n));
  return () => g(tf.tensor(x)).arraySync();
});

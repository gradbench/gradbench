import type { Runs } from "@gradbench/common/protocol.ts";
import { multipleRuns } from "@gradbench/common/util.ts";
import * as tf from "@tensorflow/tfjs";

const f = (x: tf.Tensor): tf.Tensor => {
  const a = x.max();
  return a.add(x.sub(a).exp().sum().log());
};

interface Input extends Runs {
  x: number[];
}

export const primal = multipleRuns(({ x }: Input) => {
  return () => f(tf.tensor(x)).arraySync();
});

export const gradient = multipleRuns(({ x }: Input) => {
  const g = tf.grad(f);
  return () => g(tf.tensor(x)).arraySync();
});

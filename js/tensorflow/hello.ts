import * as tf from "@tensorflow/tfjs";

const f = (x: tf.Tensor): tf.Tensor => x.square();

export const square = (x: number) => {
  return { output: f(tf.scalar(x)).arraySync() };
};

export const double = (x: number) => {
  const g = tf.grad(f);
  return { output: g(tf.scalar(x)).arraySync() };
};

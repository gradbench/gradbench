import type { Runs, Timing } from "./protocol.ts";

export const multipleRuns =
  <A extends Runs, B>(f: (input: A) => () => B) =>
  (input: A): { output: B; timings: Timing[] } => {
    const evaluate = f(input);
    const { min_runs, min_seconds } = input;
    let output;
    const timings = [];
    let elapsed = 0;
    do {
      const start = process.hrtime.bigint();
      output = evaluate();
      const end = process.hrtime.bigint();
      const nanoseconds = Number(end - start);
      timings.push({ name: "evaluate", nanoseconds });
      elapsed += nanoseconds / 1e9;
    } while (timings.length < min_runs || elapsed < min_seconds);
    return { output, timings };
  };

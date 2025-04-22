import { exec } from "node:child_process";
import { promisify } from "node:util";

export const multipleRuns = (f) => (input) => {
  const evaluate = f(input);
  const { min_runs, min_seconds } = input;
  let output;
  const timings = [];
  let elapsed = 0;
  while (timings.length < min_runs || elapsed < min_seconds) {
    const start = process.hrtime.bigint();
    output = evaluate();
    const end = process.hrtime.bigint();
    const nanoseconds = Number(end - start);
    timings.push({ name: "evaluate", nanoseconds });
    elapsed += nanoseconds / 1e9;
  }
  return { output, timings };
};

const run = promisify(exec);

export const sh = async (cmd) => (await run(cmd, { encoding: null })).stdout;

export const accommodate = (memory, bytes) => {
  const pages = Math.ceil((bytes - memory.buffer.byteLength) / 65536);
  memory.grow(pages);
};

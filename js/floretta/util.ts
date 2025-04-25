import { exec } from "node:child_process";
import { promisify } from "node:util";
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

const run = promisify(exec);

export const sh = async (cmd: string): Promise<Buffer> =>
  (await run(cmd, { encoding: null })).stdout;

export const getExports = <T>(
  module: WebAssembly.WebAssemblyInstantiatedSource,
): T => module.instance.exports as T;

export const accommodate = (
  memory: WebAssembly.Memory,
  bytes: number,
): void => {
  const pages = Math.ceil((bytes - memory.buffer.byteLength) / 65536);
  memory.grow(pages);
};

import { exec } from "node:child_process";
import { promisify } from "node:util";

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

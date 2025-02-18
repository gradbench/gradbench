import fs from "node:fs/promises";
import * as readline from "node:readline";

const shape = (matrix) => [matrix.length, matrix[0].length];

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false,
});
let memory;
let cost;
for await (const line of rl) {
  const message = JSON.parse(line);
  switch (message.kind) {
    case "define": {
      if (message.module === "kmeans") {
        const wasm = await fs.readFile(
          process.argv[2] === "--opt"
            ? "tools/floretta/kmeans-opt.wasm"
            : "tools/floretta/kmeans.wasm",
        );
        const module = await WebAssembly.instantiate(wasm);
        ({ memory, cost } = module.instance.exports);
        console.log(JSON.stringify({ id: message.id, success: true }));
      } else {
        console.log(JSON.stringify({ id: message.id, success: false }));
      }
      break;
    }
    case "evaluate": {
      const accommodate = (bytes) => {
        const pages = Math.ceil((bytes - memory.buffer.byteLength) / (2 << 15));
        if (pages > 0) {
          memory.grow(pages);
        }
      };
      const store = (offset, matrix) => {
        const [rows, cols] = shape(matrix);
        accommodate(offset + rows * cols * 8);
        const array = new Float64Array(memory.buffer, offset);
        let i = 0;
        for (const row of matrix) {
          for (const value of row) {
            array[i++] = value;
          }
        }
        return offset + 8 * i;
      };
      const input = message.input;
      const [k, d] = shape(input.centroids);
      const [n] = shape(input.points);
      const c = 0;
      const p = store(c, input.centroids);
      store(p, input.points);
      const start = process.hrtime.bigint();
      const output = cost(d, k, n, c, p);
      console.log(
        JSON.stringify({
          id: message.id,
          success: true,
          output,
          timings: [
            {
              name: "evaluate",
              nanoseconds: Number(process.hrtime.bigint() - start),
            },
          ],
        }),
      );
      break;
    }
    default: {
      console.log(JSON.stringify({ id: message.id }));
    }
  }
}
rl.close();

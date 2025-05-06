import { main } from "@gradbench/common/util.ts";
import { parseArgs } from "node:util";

const { values } = parseArgs({ options: { precision: { type: "string" } } });
let getModule = async (name: string) => await import(`./${name}.ts`);
if (values.precision !== "single") {
  getModule = () => {
    throw Error("TensorFlow.js only supports single-precision floating point.");
  };
}
main({ getModule });

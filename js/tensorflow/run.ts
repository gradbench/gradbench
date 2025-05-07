import { main } from "@gradbench/common/util-node.ts";

main({ getModule: async (name: string) => await import(`./${name}.ts`) });

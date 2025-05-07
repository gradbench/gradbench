import { main } from "@gradbench/common/util.ts";

main({ getModule: async (name: string) => await import(`./${name}.ts`) });

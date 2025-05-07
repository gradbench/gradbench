import type { Message } from "@gradbench/common/protocol.ts";
import { stdin } from "node:process";
import * as readline from "node:readline";
import { respond } from "./util.ts";

export const main = async ({
  getModule,
}: {
  getModule: (name: string) => Promise<any>;
}) => {
  for await (const line of readline.createInterface({ input: stdin })) {
    const message: Message = JSON.parse(line);
    const response = await respond({ message, getModule });
    console.log(JSON.stringify(response));
  }
};

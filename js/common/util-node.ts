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
    let response: any;
    try {
      response = await respond({ message, getModule });
    } catch (err: any) {
      let error = `${err.stack}`;
      if (!error.includes(`${err}`)) error = `${err}\n${error}`;
      response = { id: message.id, success: false, error };
    }
    console.log(JSON.stringify(response));
  }
};

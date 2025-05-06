import type {
  DefineMessage,
  DefineResponse,
  EvaluateMessage,
  EvaluateResponse,
  Message,
} from "@gradbench/common/protocol.ts";
import { stdin } from "node:process";
import * as readline from "node:readline";

const catchErrors = async <T extends any[]>(
  f: (...args: T) => Promise<{ success: boolean }>,
  ...args: T
): Promise<{ success: boolean; error?: string }> => {
  try {
    return await f(...args);
  } catch (error: any) {
    return { success: false, error: `${error.stack}` };
  }
};

const getModule = async (name: string) => await import(`./${name}.ts`);

const define = async (
  message: DefineMessage,
): Promise<Omit<DefineResponse, "id">> => {
  await getModule(message.module);
  return { success: true };
};

const evaluate = async (
  message: EvaluateMessage,
): Promise<Omit<EvaluateResponse, "id">> => {
  const module = await getModule(message.module);
  const response = await module[message.function](message.input);
  return { success: true, ...response };
};

for await (const line of readline.createInterface({ input: stdin })) {
  const message: Message = JSON.parse(line);
  const response = { id: message.id };
  switch (message.kind) {
    case "define": {
      Object.assign(response, await catchErrors(define, message));
      break;
    }
    case "evaluate": {
      Object.assign(response, await catchErrors(evaluate, message));
      break;
    }
  }
  console.log(JSON.stringify(response));
}

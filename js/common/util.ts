import type {
  DefineMessage,
  DefineResponse,
  EvaluateMessage,
  EvaluateResponse,
  Message,
} from "@gradbench/common/protocol.ts";
import { stdin } from "node:process";
import * as readline from "node:readline";
import type { Runs, Timing } from "./protocol.ts";

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

export const main = async ({
  getModule,
}: {
  getModule: (name: string) => Promise<any>;
}) => {
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
};

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

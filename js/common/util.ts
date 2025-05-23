import type {
  Base,
  DefineMessage,
  DefineResponse,
  EvaluateMessage,
  EvaluateResponse,
  Message,
} from "@gradbench/common/protocol.ts";
import type { Runs, Timing } from "./protocol.ts";

const catchErrors = async <T extends any[]>(
  f: (...args: T) => Promise<{ success: boolean }>,
  ...args: T
): Promise<{ success: boolean; error?: string }> => {
  try {
    return await f(...args);
  } catch (err: any) {
    let error = `${err.stack}`;
    // In Firefox, the `stack` field doesn't contain the actual error message.
    if (!error.includes(`${err}`)) error = `${err}\n${error}`;
    return { success: false, error };
  }
};

export const respond = async ({
  message,
  getModule,
}: {
  message: Message;
  getModule: (name: string) => Promise<any>;
}): Promise<Base | DefineResponse | EvaluateResponse> => {
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
  return response;
};

export const multipleRuns =
  <A extends Runs, B>(f: (input: A) => () => B) =>
  (input: A): { output: B; timings: Timing[] } => {
    const evaluate = f(input);
    const { min_runs, min_seconds } = input;
    let output;
    const timings = [];
    let seconds = 0;
    do {
      const start = performance.now();
      output = evaluate();
      const end = performance.now();
      const milliseconds = end - start;
      const nanoseconds = Math.round(milliseconds * 1e6);
      timings.push({ name: "evaluate", nanoseconds });
      seconds += milliseconds / 1e3;
    } while (timings.length < min_runs || seconds < min_seconds);
    return { output, timings };
  };

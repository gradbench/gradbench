import { stdin } from "node:process";
import * as readline from "node:readline";

const catchErrors = async (f, ...args) => {
  try {
    return await f(...args);
  } catch (error) {
    return { success: false, error: `${error.stack}` };
  }
};

const getModule = async (name) => await import(`./${name}.js`);

const define = async (message) => {
  await getModule(message.module);
  return { success: true };
};

const evaluate = async (message) => {
  const module = await getModule(message.module);
  const response = await module[message.function](message.input);
  return { success: true, ...response };
};

for await (const line of readline.createInterface({ input: stdin })) {
  const message = JSON.parse(line);
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

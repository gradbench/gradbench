import { Message } from "@gradbench/common/protocol.ts";
import { respond } from "@gradbench/common/util.ts";

const getModule = async (name: string) => await import(`./${name}.ts`);

const textArea = document.getElementById("textarea") as HTMLTextAreaElement;

document.getElementById("button")!.addEventListener("click", async () => {
  for (const line of textArea.value.split("\n")) {
    const message: Message = JSON.parse(line);
    const response = await respond({ message, getModule });
    console.log(response);
  }
});

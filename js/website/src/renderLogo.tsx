import { renderToString } from "react-dom/server";
import { Logo } from "./Logo.tsx";

console.log(renderToString(<Logo colors={["#3e4756"]} />));

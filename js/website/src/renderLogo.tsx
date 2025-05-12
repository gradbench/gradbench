import { renderToString } from "react-dom/server";
import { Logo } from "./components/Logo";

console.log(renderToString(<Logo colors={["#3e4756"]} />));

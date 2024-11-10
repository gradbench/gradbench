import { Fragment } from "react/jsx-runtime";
import "./App.css";
import raw from "./summary.json?raw";

interface Cell {
  tool: string;
  status: "implemented" | "nonimplemented";
}

interface Row {
  eval: string;
  tools: Cell[];
}

interface Summary {
  table: Row[];
}

const Table = ({ date }: { date: string }) => {
  const tag = `nightly-${date}`;
  const summary: Summary = JSON.parse(raw);
  const numTools = summary.table[0].tools.length;
  const cellSize = "30px";
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `min-content repeat(${numTools}, ${cellSize})`,
        gridTemplateRows: `min-content repeat(${numTools}, ${cellSize})`,
        gap: "3px",
      }}
    >
      <div />
      {summary.table[0].tools.map((cell) => (
        <div key={cell.tool} className="column-header">
          <a
            href={`https://github.com/gradbench/gradbench/tree/${tag}/tools/${cell.tool}`}
          >
            {cell.tool}
          </a>
        </div>
      ))}
      {summary.table.map((row) => (
        <Fragment key={row.eval}>
          <div className="row-header">
            <a
              href={`https://github.com/gradbench/gradbench/tree/${tag}/evals/${row.eval}`}
            >
              {row.eval}
            </a>
          </div>
          {row.tools.map((cell) => (
            <div key={cell.tool} className="cell">
              {cell.status === "implemented" ? "✓" : ""}
            </div>
          ))}
        </Fragment>
      ))}
    </div>
  );
};

const App = () => {
  const today = new Date().toISOString().split("T")[0];
  return (
    <>
      <h1>
        <a href="https://github.com/gradbench/gradbench">GradBench</a>{" "}
        <input type="date" min="2024-11-04" max={today} value={today} />
      </h1>
      <Table date={today} />
    </>
  );
};

export default App;

import { useEffect, useState } from "react";
import { Fragment } from "react/jsx-runtime";
import "./App.css";

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
  const [summary, setSummary] = useState<Summary | undefined>(undefined);

  useEffect(() => {
    const download = async () => {
      const url = `https://github.com/gradbench/gradbench/releases/download/nightly-${date}/summary.json`;
      const response = await fetch(url);
      setSummary(await response.json());
    };
    download();
  }, [date]);

  if (summary === undefined) return <p>Downloading...</p>;

  const tag = `nightly-${date}`;
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
        <input type="date" defaultValue={today} />
      </h1>
      <Table date={today} />
    </>
  );
};

export default App;

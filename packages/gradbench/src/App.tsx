import { useEffect, useState } from "react";
import { Fragment } from "react/jsx-runtime";
import "./App.css";

interface Cell {
  tool: string;
  status: "unimplemented" | "incorrect" | "correct";
}

interface Row {
  eval: string;
  tools: Cell[];
}

interface Summary {
  table: Row[];
}

const Table = ({ date }: { date: string }) => {
  const [summary, setSummary] = useState<undefined | null | Summary>(undefined);

  useEffect(() => {
    setSummary(undefined);
    const download = async () => {
      try {
        const url = `https://raw.githubusercontent.com/gradbench/gradbench/refs/tags/nightly-${date}/summary.json`;
        const response = await fetch(url);
        setSummary(await response.json());
      } catch (e) {
        setSummary(null);
      }
    };
    download();
  }, [date]);

  if (summary === undefined) return <p>Downloading...</p>;
  if (summary === null) return <p>No data found for {date}.</p>;

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
          {cell.tool}
        </div>
      ))}
      {summary.table.map((row) => (
        <Fragment key={row.eval}>
          <div className="row-header">{row.eval}</div>
          {row.tools.map((cell) => {
            switch (cell.status) {
              case "incorrect": {
                return (
                  <div key={cell.tool} className="cell incorrect">
                    ✗
                  </div>
                );
              }
              case "correct": {
                return (
                  <div key={cell.tool} className="cell correct">
                    ✓
                  </div>
                );
              }
              default: {
                return (
                  <div key={cell.tool} className="cell unimplemented"></div>
                );
              }
            }
          })}
        </Fragment>
      ))}
    </div>
  );
};

const today = new Date().toISOString().split("T")[0];

const App = () => {
  const [date, setDate] = useState(today);
  return (
    <>
      <h1>
        <a href="https://github.com/gradbench/gradbench">GradBench</a>{" "}
        <input
          type="date"
          defaultValue={date}
          onChange={(e) => setDate(e.target.value)}
        />
      </h1>
      <Table date={date} />
    </>
  );
};

export default App;

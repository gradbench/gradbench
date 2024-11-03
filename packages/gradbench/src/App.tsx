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
  return (
    <table>
      <thead>
        <tr>
          <th></th>
          {summary.table[0].tools.map((cell) => (
            <th scope="col">
              <a
                href={`https://github.com/gradbench/gradbench/tree/${tag}/tools/${cell.tool}`}
              >
                {cell.tool}
              </a>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {summary.table.map((row) => (
          <tr>
            <th scope="row">
              <a
                href={`https://github.com/gradbench/gradbench/tree/${tag}/evals/${row.eval}`}
              >
                {row.eval}
              </a>
            </th>
            {row.tools.map((cell) => (
              <td>{cell.status === "implemented" ? "✓" : ""}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
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

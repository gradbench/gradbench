import { Row, Summary } from "../store.ts";
import { Fragment } from "react";

const formatDuration = (duration: number): [string, string] => {
  if (duration < 1e-4) {
    return [(duration * 1e6).toPrecision(2).replace("0.", "."), "µs"];
  }
  if (duration < 1e-1) {
    return [(duration * 1e3).toPrecision(2).replace("0.", "."), "ms"];
  }
  if (duration < 1e2) {
    return [duration.toPrecision(2).replace("0.", "."), "s"];
  }
  return [duration.toFixed(0), "s"];
};

const ScoredRow = ({ tools }: Row) => {
  const maxScore = Math.max(
    ...tools.flatMap(({ score }) => (score === undefined ? [] : [score])),
  );
  return tools.map(({ tool, outcome, score, status }) => {
    if (score !== undefined) {
      const lightness = 100 - 50 * (score / maxScore);
      const [duration, unit] = formatDuration(1 / score);
      return (
        <div
          key={tool}
          className="cell"
          style={{
            backgroundColor: `hsl(240 100% ${lightness}%)`,
            color: lightness < 70 ? "#e2e2ff" : "#0d0d1a"
          }}
        >
          <span>
            <span className="cell__duration">{duration}</span>
            <span className="cell__unit">{unit}</span>
          </span>
        </div>
      );
    } else if (outcome !== undefined && outcome !== "undefined") {
      // This means the tool was defined for this eval but had an unsuccessful
      // outcome, like `timeout`.
      const alpha = 50;
      return (
        <div
          key={tool}
          className="cell"
          style={{ backgroundColor: `rgb(255 255 255 / ${alpha}` }}
          title={outcome}
        >
          {(outcome === "error" || outcome == "failure" || outcome == "interrupt" || outcome == "invalid") &&
            <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#000000"><path d="M479.79-298.77q11.94 0 20.23-8.08 8.29-8.08 8.29-20.02t-8.08-20.23q-8.08-8.28-20.02-8.28t-20.23 8.07q-8.29 8.08-8.29 20.02t8.08 20.23q8.08 8.29 20.02 8.29ZM454-428.92h52v-240h-52v240ZM480.34-116q-75.11 0-141.48-28.42-66.37-28.42-116.18-78.21-49.81-49.79-78.25-116.09Q116-405.01 116-480.39q0-75.38 28.42-141.25t78.21-115.68q49.79-49.81 116.09-78.25Q405.01-844 480.39-844q75.38 0 141.25 28.42t115.68 78.21q49.81 49.79 78.25 115.85Q844-555.45 844-480.34q0 75.11-28.42 141.48-28.42 66.37-78.21 116.18-49.79 49.81-115.85 78.25Q555.45-116 480.34-116Zm-.34-52q130 0 221-91t91-221q0-130-91-221t-221-91q-130 0-221 91t-91 221q0 130 91 221t221 91Zm0-312Z"/></svg>
          }
          {outcome === "timeout" &&
            <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#000000"><path d="M480-510.61q67.31 0 113.58-47.04 46.27-47.04 46.27-114.35v-120h-319.7v120q0 67.31 46.27 114.35 46.27 47.04 113.58 47.04ZM212-116v-52h56.16v-120q0-65.54 36.46-117.96 36.46-52.43 96.46-74.04-59-22-95.96-74.23-36.96-52.23-36.96-117.77v-120H212v-52h536v52h-56.16v120q0 65.54-36.96 117.77Q617.92-502 558.92-480q60 21.61 96.46 74.04 36.46 52.42 36.46 117.96v120H748v52H212Z"/></svg>
          }
        </div>
      );
    }
    switch (status) {
      case "incorrect": {
        return (
          <div key={tool} className="cell incorrect">
            ✗
          </div>
        );
      }
      case "correct": {
        return (
          <div key={tool} className="cell correct">
            ✓
          </div>
        );
      }
      default: {
        return <div key={tool} className="cell" />;
      }
    }
  });
};

interface SummaryVizProps {
  summary: Summary;
  activeEval: string | null;
  onActiveEvalChange: (activeEval: string | null) => void;
}

const SummaryViz = ({ summary, activeEval, onActiveEvalChange }: SummaryVizProps) => {
  const numEvals = summary.table.length;
  const numTools = summary.table[0].tools.length;
  const cellSize = "30px";
  return (
    <>
      <ul>
        <li>
          A <em>grey</em> cell means the tool did not successfully complete that
          eval.
        </li>
        <li>
          A <em>white</em> cell means the tool is <strong>slow</strong> for that
          eval.
        </li>
        <li>
          A <em>blue</em> cell means the tool is <strong>fast</strong> for that
          eval.
        </li>
      </ul>
      <div
        className="table"
        style={{
          display: "grid",
          gridTemplateColumns: `min-content repeat(${numTools}, ${cellSize})`,
          gridTemplateRows: `min-content repeat(${numEvals}, ${cellSize})`,
          gap: "5px",
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
            <div
              className={`row-header ${summary.version === 1 ? "header-clickable" : ""}`}
              onClick={() => onActiveEvalChange(row.eval)}
            >
              {row.eval}
            </div>
            <ScoredRow {...row} />
          </Fragment>
        ))}
      </div>
    </>
  );
};

export default SummaryViz;

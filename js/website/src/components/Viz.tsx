import { useState } from "react";
import { Row, Summary } from "../summary.ts";
import { Fragment } from "react";
import Stats from "./Stats.tsx";

const formatScore = (score: number): string => {
  return (Math.round(score * 1e2) / 1e2).toString().replace("0.", ".");
};

const ScoredRow = ({ tools }: Row) => {
  const maxScore = Math.max(
    ...tools.flatMap(({ score }) => (score === undefined ? [] : [score])),
  );
  return tools.map(({ tool, outcome, score, status }) => {
    if (score !== undefined) {
      const lightness = 100 - 50 * (score / maxScore);
      return (
        <div
          key={tool}
          className="cell"
          style={{
            backgroundColor: `hsl(240 100% ${lightness}%)`,
            color: lightness < 70 ? "#e2e2ff" : "#0d0d1a"
          }}
        >{formatScore(score / maxScore)}</div>
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
          {outcome === "error" &&
            <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#000000"><path d="M479.79-298.77q11.94 0 20.23-8.08 8.29-8.08 8.29-20.02t-8.08-20.23q-8.08-8.28-20.02-8.28t-20.23 8.07q-8.29 8.08-8.29 20.02t8.08 20.23q8.08 8.29 20.02 8.29ZM454-428.92h52v-240h-52v240ZM480.34-116q-75.11 0-141.48-28.42-66.37-28.42-116.18-78.21-49.81-49.79-78.25-116.09Q116-405.01 116-480.39q0-75.38 28.42-141.25t78.21-115.68q49.79-49.81 116.09-78.25Q405.01-844 480.39-844q75.38 0 141.25 28.42t115.68 78.21q49.81 49.79 78.25 115.85Q844-555.45 844-480.34q0 75.11-28.42 141.48-28.42 66.37-78.21 116.18-49.79 49.81-115.85 78.25Q555.45-116 480.34-116Zm-.34-52q130 0 221-91t91-221q0-130-91-221t-221-91q-130 0-221 91t-91 221q0 130 91 221t221 91Zm0-312Z"/></svg>
          }
          {outcome === "timeout" &&
            <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#000000"><path d="M367.69-826v-52h224.62v52H367.69ZM454-391.69h52v-224.62h-52v224.62ZM480-116q-66.15 0-123.42-24.78-57.26-24.77-100.15-67.65-42.88-42.89-67.65-100.15Q164-365.85 164-432t24.78-123.42q24.77-57.26 67.65-100.15 42.89-42.88 100.15-67.65Q413.85-748 479.56-748q56.36 0 108.59 19.89 52.23 19.88 95.93 54.88l45.84-44.84 36.15 36.15-44.84 45.84q35 42.7 54.88 95.16Q796-488.46 796-431.86q0 66.01-24.78 123.28-24.77 57.26-67.65 100.15-42.89 42.88-100.15 67.65Q546.15-116 480-116Zm0-52q110 0 187-77t77-187q0-110-77-187t-187-77q-110 0-187 77t-77 187q0 110 77 187t187 77Zm0-264Z"/></svg>
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

const Viz = ({ prefix, summary }: { prefix: string; summary: Summary }) => {
  const [activeEval, setActiveEval] = useState<string | undefined>(undefined);
  const numEvals = summary.table.length;
  const numTools = summary.table[0].tools.length;
  const cellSize = "30px";
  // We use the URL as the `key` for the `Stats` component so that its state
  // completely resets when the URL changes; that way, when that component
  // fetches the data, it doesn't need to check before overwriting its state.
  const url =
    activeEval === undefined
      ? undefined
      : summary.version === 1
        ? `${prefix}/evals/${activeEval}/summary.json`
        : undefined;
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
              onClick={() => {
                setActiveEval(row.eval);
              }}
            >
              {row.eval}
            </div>
            <ScoredRow {...row} />
          </Fragment>
        ))}
      </div>
      {url === undefined ? (
        <></>
      ) : (
        <>
          <h2>{activeEval}</h2>
          <Stats key={url} url={url} />
        </>
      )}
    </>
  );
};

export default Viz;

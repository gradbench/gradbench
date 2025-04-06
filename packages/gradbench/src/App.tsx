import { useEffect, useRef, useState } from "react";
import { Fragment } from "react/jsx-runtime";
import "./App.css";
import { Stats } from "./Stats.tsx";

/** Return a YYYY-MM-DD date string from a `Date` object. */
const dateString = (date: Date): string => date.toISOString().split("T")[0];

/**
 * Return `date` if it is a valid YYYY-MM-DD date string, otherwise `undefined`.
 */
const parseDate = (date: string | null | undefined): string | undefined => {
  if (date === null || date === undefined) return undefined;
  try {
    if (dateString(new Date(date)) === date) {
      return date;
    }
  } catch (_) {
    return undefined;
  }
};

/** Return the URL prefix we should download from. */
const urlPrefix = (params: {
  commit: string | undefined;
  date: string | undefined;
}): string => {
  const prefix = "https://raw.githubusercontent.com/gradbench/gradbench";
  if (params.date !== undefined) {
    return `${prefix}/refs/tags/nightly-${params.date}`;
  } else if (params.commit !== undefined) {
    return `${prefix}/${params.commit}`;
  } else {
    return `${prefix}/refs/heads/ci/refs/heads/nightly`;
  }
};

interface Cell {
  tool: string;
  outcome?:
    | "interrupt"
    | "timeout"
    | "invalid"
    | "failure"
    | "undefined"
    | "error";
  score?: number;
  status?: "unimplemented" | "incorrect" | "correct";
}

interface Row {
  eval: string;
  tools: Cell[];
}

interface Summary {
  version?: number;
  date?: string;
  table: Row[];
}

/** Attempt to download the summary using the given URL prefix. */
const download = async (prefix: string): Promise<Summary | undefined> => {
  try {
    const response = await fetch(`${prefix}/summary.json`);
    return await response.json();
  } catch (_) {
    return undefined;
  }
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
          style={{ backgroundColor: `hsl(240 100% ${lightness}%)` }}
        />
      );
    } else if (outcome !== undefined && outcome !== "undefined") {
      // This means the tool was defined for this eval but had an unsuccessful
      // outcome, like `timeout`.
      const alpha = 50;
      return (
        <div
          key={tool}
          className="cell"
          style={{ backgroundColor: `rgb(255 255 255 / ${alpha}%)` }}
        />
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

const randomColor = () => {
  return `#${Math.floor(Math.random() * 16777215).toString(16)}`;
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

interface DownloadedSummary {
  /** The URL prefix we used to download the summary. */
  prefix: string;

  /** The summary we downloaded, if there was one. */
  summary: Summary | undefined;
}

interface State {
  /** The date that we are currently displaying, if any. */
  date: string | undefined;

  /** The summary that we are currently displaying, if any. */
  summary: DownloadedSummary | undefined;
}

const App = () => {
  const params = new URL(window.location.href).searchParams;
  const commit: string | undefined = params.get("commit") ?? undefined;
  const [state, setState] = useState<State>({
    date: parseDate(params.get("date")),
    summary: undefined,
  });
  const prefix = urlPrefix({ commit, date: state.date });
  const downloaded = state.summary?.prefix;
  const logoRef = useRef<HTMLObjectElement>(null);

  // Change color when the logo has loaded
  const handleLogoLoad = () => {
    const object = logoRef.current;
    const logo = object!.contentDocument;

    const gradient = logo!.querySelector("#bggradient");
    const stops = gradient!.querySelectorAll("stop");
    stops[0].setAttribute("stop-color", randomColor());
    stops[1].setAttribute("stop-color", randomColor());
  };

  useEffect(() => {
    // Nothing to do if we've already downloaded this summary.
    if (prefix === downloaded) return;
    (async () => {
      const summary = await download(prefix);
      setState((current) => {
        // Only overwrite the summary if the URL prefix we're using is still the
        // same as the one wanted by the current state.
        if (urlPrefix({ commit, date: current.date }) !== prefix)
          return current;
        const newState: State = { ...current, summary: { prefix, summary } };
        if (current.date === undefined) {
          const date = parseDate(summary?.date);
          // If the user hasn't picked a date and there's no `commit` in the
          // query parameters, then we just downloaded the summary for the most
          // recent nightly build, which probably has a date, so we can just
          // store that to show in the date picker.
          if (date !== undefined) {
            newState.date = date;
          }
        }
        return newState;
      });
    })();

    const logo = logoRef.current;

    if (logo) {
      logo.onload = handleLogoLoad;
    }
  }, [commit, prefix, downloaded]);
  const pickDate = (date: string) => {
    if (parseDate(date) === null) return;
    const url = new URL(window.location.href);
    url.searchParams.set("date", date);
    window.history.pushState(null, "", url.href);
    setState((current) => ({ ...current, date }));
  };
  return (
    <>
      <object
        id="logo"
        ref={logoRef}
        type="image/svg+xml"
        data="/src/logo.svg"
      />
      <h1>
        <a href="https://github.com/gradbench/gradbench">GradBench</a>{" "}
      </h1>
      <nav>
        <button
          disabled={state.date === undefined}
          onClick={() => {
            if (state.date === undefined) return;
            const d = new Date(state.date);
            d.setDate(d.getDate() - 1);
            pickDate(dateString(d));
          }}
        >
          ◀
        </button>{" "}
        <input
          type="date"
          value={state.date ?? ""}
          onChange={(e) => pickDate(e.target.value)}
        />{" "}
        <button
          disabled={state.date === undefined}
          onClick={() => {
            if (state.date === undefined) return;
            const d = new Date(state.date);
            d.setDate(d.getDate() + 1);
            pickDate(dateString(d));
          }}
        >
          ▶
        </button>
      </nav>
      {state.summary === undefined ? (
        <p>Downloading...</p>
      ) : state.summary.summary === undefined ? (
        <p>
          No data found
          {state.date !== undefined ? (
            ` for ${state.date}`
          ) : commit !== undefined ? (
            <>
              {" "}
              at commit <code>{commit}</code>
            </>
          ) : (
            " for the latest nightly build"
          )}
          .
        </p>
      ) : (
        <Viz prefix={prefix} summary={state.summary.summary} />
      )}
    </>
  );
};

export default App;

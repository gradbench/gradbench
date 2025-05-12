import { useEffect, useState } from "react";
import "../css/App.css";
import Logo from "./Logo.tsx";
import Viz from "./Viz.tsx";
import { randomColor, dateString, parseDate } from "../utils.ts";
import { Summary, downloadSummary } from "../summary.ts";

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
  useEffect(() => {
    // Nothing to do if we've already downloaded this summary.
    if (prefix === downloaded) return;
    (async () => {
      const summary = await downloadSummary(prefix);
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
      <div className="logo">
        <Logo colors={[randomColor(), randomColor()]} />
      </div>
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

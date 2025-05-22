import { useEffect, useState } from "react";
import { useParam } from "../hooks/useParam.ts";
import {
  EvalStats,
  NotFoundError,
  Summary,
  downloadEvalStat,
  downloadSummary,
} from "../store.ts";
import EvalViz from "./EvalViz.tsx";
import Header from "./Header.tsx";
import SummaryViz from "./SummaryViz.tsx";

enum Status {
  Loading = 0,
  Error = 1,
  NotFound = 2,
  Ok = 3,
}

const App = () => {
  const [date, setDate] = useParam("date", null);
  const [commit] = useParam("commit", null);
  const [activeEval, setActiveEval] = useState<string | null>(null);

  const [summary, setSummary] = useState<Summary | null>(null);
  const [summaryStatus, setSummaryStatus] = useState(Status.Loading);
  useEffect(() => {
    setSummaryStatus(Status.Loading);
    downloadSummary(date, commit)
      .then((summary) => {
        // A race condition can happen if successive downloadSummary call
        // returns out of order. Then select summary will be the last recived.
        // Though this is rare enough to pretend this does not happens
        setSummary(summary);
        // Here we call setDate while date is dependency of the useEffect hook.
        // This will trigger a useless re-run of the useEffect. Though the
        // preformance overhead is minimal.
        if (summary.date) setDate(summary.date);
        setSummaryStatus(Status.Ok);
      })
      .catch((err) => {
        if (err instanceof NotFoundError) {
          setSummaryStatus(Status.NotFound);
        } else {
          console.error(err);
          setSummaryStatus(Status.Error);
        }
      });
  }, [date, commit, setDate]);

  const [evalStats, setEvalStats] = useState<EvalStats | null>(null);
  const [evalStatsStatus, setEvalStatsStatus] = useState(Status.Loading);
  useEffect(() => {
    if (activeEval === null) return;
    if (activeEval === "hello") {
      setEvalStats(null);
      setEvalStatsStatus(Status.Ok);
      return;
    }
    setEvalStatsStatus(Status.Loading);
    downloadEvalStat(date, commit, activeEval)
      .then((evalStats) => {
        // A race condition can happen if successive downloadEvalStat call
        // returns out of order. Then select evalStats will be the last recived.
        // Though this is rare enough to pretend this does not happens
        setEvalStats(evalStats);
        setEvalStatsStatus(Status.Ok);
      })
      .catch((err) => {
        console.error(err);
        setEvalStatsStatus(Status.Error);
      });
  }, [date, commit, activeEval]);

  return (
    <>
      <Header date={date} onDateChange={setDate} />

      <section className="section">
        {summaryStatus === Status.Loading && (
          <p>Downloading the summary graph...</p>
        )}
        {summaryStatus === Status.NotFound && date !== null && (
          <p>No data found for date {date}</p>
        )}
        {summaryStatus === Status.NotFound &&
          date === null &&
          commit !== null && <p>No data found for commit {commit}</p>}
        {summaryStatus === Status.NotFound &&
          date === null &&
          commit === null && <p>No data found</p>}
        {summaryStatus === Status.Error && (
          <p>Could not download the summary (error dumped in console)</p>
        )}
        {summaryStatus === Status.Ok && summary !== null && (
          <SummaryViz
            summary={summary as Summary}
            activeEval={activeEval}
            onActiveEvalChange={setActiveEval}
          />
        )}
      </section>

      <section className="section">
        {evalStatsStatus === Status.Loading && activeEval !== null && (
          <p>Downloading the eval stats...</p>
        )}
        {evalStatsStatus === Status.Error && activeEval !== null && (
          <p>Could not download the eval stats (error dumped in console)</p>
        )}
        {evalStatsStatus === Status.Ok &&
          activeEval !== null &&
          activeEval !== "hello" &&
          evalStats !== null && (
            <EvalViz
              activeEval={activeEval}
              evalStats={evalStats as EvalStats}
            />
          )}
      </section>
    </>
  );
};

export default App;

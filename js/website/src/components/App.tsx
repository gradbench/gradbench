import { useEffect, useState } from "react";
import { NotFoundError, Summary, downloadEvalStat, downloadSummary } from "../store.ts";
import { useParam } from "../hooks/useParam.ts";
import { EvalStats } from "../store.ts";
import Header from "./Header.tsx";
import SummaryViz from "./SummaryViz.tsx";
import EvalViz from "./EvalViz.tsx";
import "../styles/App.css";

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
    // NOTE: Unnecessary calls to downloadSummary will happen but the already
    // cumbersome enough without takeing care of that
    setSummaryStatus(Status.Loading);
    downloadSummary(date, commit)
      .then(summary => {
        // NOTE: Let's hope the summaries arrive in order 🤞
        setSummary(summary);
        if (summary.date) setDate(summary.date);
        setSummaryStatus(Status.Ok);
      })
      .catch(err => {
        if (err instanceof NotFoundError) {
          setSummaryStatus(Status.NotFound);
        } else {
          console.error(err);
          setSummaryStatus(Status.Error);
        }
      });
  }, [date, commit]);

  const [evalStats, setEvalStats] = useState<EvalStats | null>(null);
  const [evalStatsStatus, setEvalStatsStatus] = useState(Status.Loading);
  useEffect(() => {
    if (activeEval === null) return;
    setEvalStatsStatus(Status.Loading);
    downloadEvalStat(date, commit, activeEval)
      .then(evalStats => {
        setEvalStats(evalStats);
        setEvalStatsStatus(Status.Ok);
      })
      .catch(err => {
        console.error(err);
        setEvalStatsStatus(Status.Error);
      });
  }, [activeEval]);

  return (
    <main>
      <Header date={date} onDateChange={setDate} />

      {summaryStatus === Status.Loading &&
        <p>Downloading the summary graph...</p>
      }
      {summaryStatus === Status.NotFound && date !== null &&
        <p>No data found for date {date}</p>
      }
      {summaryStatus === Status.NotFound && date === null && commit !== null &&
        <p>No data found for commit {commit}</p>
      }
      {summaryStatus === Status.NotFound && date === null && commit === null &&
        <p>No data found</p>
      }
      {summaryStatus === Status.Error &&
        <p>Could not download the summary (error dumped in console)</p>
      }
      {summaryStatus === Status.Ok &&
        <SummaryViz summary={summary as Summary} activeEval={activeEval} onActiveEvalChange={setActiveEval} />
      }

      {activeEval !== null && evalStatsStatus === Status.Loading &&
        <p>Downloading the eval stats...</p>
      }
      {activeEval !== null && evalStatsStatus === Status.Error &&
        <p>Could not download the eval stats (error dumped in console)</p>
      }
      {activeEval !== null && evalStatsStatus === Status.Ok &&
        <EvalViz activeEval={activeEval} evalStats={evalStats as EvalStats} />
      }
    </main>
  )
}

export default App;

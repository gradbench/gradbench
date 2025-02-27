import { useEffect, useState } from "react";
import { VegaLite } from "react-vega";
import { TopLevelSpec } from "vega-lite";

interface Duration {
  secs: number;
  nanos: number;
}

const seconds = (duration: Duration): number =>
  duration.secs + duration.nanos / 1e9;

interface Durations {
  primal: Duration;
  derivative: Duration;
}

interface Stats {
  tools: Record<string, Record<string, Durations>>;
}

/**
 * Attempt to download stats from the given URL.
 */
const downloadStats = async (url: string): Promise<Stats | undefined> => {
  try {
    const response = await fetch(url);
    return await response.json();
  } catch (_) {
    return undefined;
  }
};

export const Stats = ({ url }: { url: string }) => {
  const [stats, setStats] = useState<Stats | undefined>(undefined);
  useEffect(() => {
    (async () => setStats(await downloadStats(url)))();
  }, [url]);
  if (stats === undefined) return <></>;
  const spec: TopLevelSpec = {
    width: 500,
    height: 250,
    mark: { type: "line", point: true },
    params: [
      {
        name: "tool",
        select: { type: "point", fields: ["tool"] },
        bind: "legend",
      },
    ],
    transform: [{ filter: { param: "tool" } }],
    encoding: {
      color: {
        field: "tool",
        type: "nominal",
        legend: { values: Object.keys(stats.tools) },
      },
      x: {
        field: "workload",
        type: "nominal",
        axis: { labelAngle: -45 },
      },
      y: {
        title: "primal seconds",
        field: "primal",
        type: "quantitative",
        scale: { type: "log" },
      },
    },
    data: { name: "table" },
  };
  const data = {
    table: Object.entries(stats.tools).flatMap(([tool, toolStats]) =>
      Object.entries(toolStats).map(([workload, evalStats]) => ({
        tool,
        workload: workload,
        primal: seconds(evalStats.primal),
        derivative: seconds(evalStats.derivative),
      })),
    ),
  };
  return (
    <>
      <p>
        Hold <kbd>Shift</kbd> and click on parts of the legend to hide and show
        different tools.
      </p>
      <VegaLite spec={spec} data={data} />
    </>
  );
};

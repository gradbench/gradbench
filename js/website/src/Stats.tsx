import { useEffect, useState } from "react";
import { PlainObject, VegaLite } from "react-vega";
import { TopLevelSpec } from "vega-lite";
import "./Stats.css";

// These colors have been determined by sampling the tool websites.
// They are not picked to make a particularly pleasing scheme, and
// some are unfortunately a little close to each other.
const colors = {
  "ad-hpp": "#f6a800",
  "adol-c": "#9da117",
  adept: "#d8702f",
  codipack: "#d02718",
  cppad: "#eeb10f",
  enzyme: "#173559",
  "enzyme-jl": "#0094e8",
  finite: "#aaaaaa",
  floretta: "#f10537",
  futhark: "#5f021f",
  haskell: "#5e5086",
  jax: "#5e98f6",
  manual: "#000000",
  ocaml: "#c24f1e",
  pytorch: "#ee4c2c",
  scilean: "#5c123a",
  tapenade: "#047f01",
  tensorflow: "#ff8d00",
  zygote: "#6daa5e",
};

const makeSpec = ({
  title,
  yaxis,
  tools,
}: {
  title: string;
  yaxis: string;
  tools: string[];
}): TopLevelSpec => {
  return {
    title,
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
        scale: {
          domain: Object.keys(colors),
          range: Object.values(colors),
        },
        legend: { values: tools },
      },
      x: {
        field: "workload",
        type: "nominal",
        sort: null,
        axis: { labelAngle: -45 },
      },
      y: {
        title: yaxis,
        field: "amount",
        type: "quantitative",
        scale: { type: "log" },
      },
    },
    data: { name: "table" },
  };
};

interface Duration {
  secs: number;
  nanos: number;
}

const seconds = (duration: Duration | undefined): number | undefined =>
  duration !== undefined ? duration.secs + duration.nanos / 1e9 : undefined;

const divide = (
  a: number | undefined,
  b: number | undefined,
): number | undefined =>
  a !== undefined && b !== undefined ? a / b : undefined;

interface Durations {
  primal?: Duration;
  derivative?: Duration;
}

interface Stats {
  tools: Record<string, Record<string, Durations>>;
}

const makeData = (
  stats: Stats,
  getAmount: (durations: Durations) => number | undefined,
): PlainObject => ({
  table: Object.entries(stats.tools).flatMap(([tool, toolStats]) =>
    Object.entries(toolStats).flatMap(([workload, evalStats]) => {
      const amount = getAmount(evalStats);
      return amount !== undefined ? [{ tool, workload, amount }] : [];
    }),
  ),
});

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
  const tools = Object.keys(stats.tools);
  return (
    <>
      <p>
        Hold <kbd>Shift</kbd> and click on parts of the legend to hide and show
        different tools.
      </p>
      <div className="chart-box">
        <VegaLite
          renderer="svg"
          spec={makeSpec({
            title: "derivative",
            yaxis: "derivative (seconds)",
            tools,
          })}
          data={makeData(stats, ({ derivative }) => seconds(derivative))}
        />
      </div>
      <div className="chart-box">
        <VegaLite
          renderer="svg"
          spec={makeSpec({
            title: "ratio",
            yaxis: "derivative / primal",
            tools,
          })}
          data={makeData(stats, ({ primal, derivative }) =>
            divide(seconds(derivative), seconds(primal)),
          )}
        />
      </div>
      <div className="chart-box">
        <VegaLite
          renderer="svg"
          spec={makeSpec({
            title: "primal",
            yaxis: "primal (seconds)",
            tools,
          })}
          data={makeData(stats, ({ primal }) => seconds(primal))}
        />
      </div>
    </>
  );
};

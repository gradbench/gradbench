export interface Cell {
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

export interface Row {
  eval: string;
  tools: Cell[];
}

export interface Summary {
  version?: number;
  date?: string;
  table: Row[];
}

export class NotFoundError extends Error {}

/** Return the URL prefix we should download from. */
const urlPrefix = (date: string | null, commit: string | null): string => {
  const prefix = "https://raw.githubusercontent.com/gradbench/gradbench";
  if (date !== null) {
    return `${prefix}/refs/tags/nightly-${date}`;
  } else if (commit !== null) {
    return `${prefix}/${commit}`;
  } else {
    return `${prefix}/refs/heads/ci/refs/heads/nightly`;
  }
};

export async function downloadSummary(
  date: string | null,
  commit: string | null,
): Promise<Summary> {
  const res = await fetch(`${urlPrefix(date, commit)}/summary.json`);
  if (res.status == 404) {
    throw new NotFoundError("summary not found");
  }
  return res.json();
}

export interface Duration {
  secs: number;
  nanos: number;
}

export interface Durations {
  primal?: Duration;
  derivative?: Duration;
}

export interface EvalStats {
  tools: Record<string, Record<string, Durations>>;
}

export async function downloadEvalStat(
  date: string | null,
  commit: string | null,
  activeEval: string | null,
): Promise<EvalStats> {
  const res = await fetch(
    `${urlPrefix(date, commit)}/evals/${activeEval}/summary.json`,
  );
  return await res.json();
}

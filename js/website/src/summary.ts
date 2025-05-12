
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

/** Attempt to download the summary using the given URL prefix. */
export const downloadSummary = async (prefix: string): Promise<Summary | undefined> => {
  try {
    const response = await fetch(`${prefix}/summary.json`);
    return await response.json();
  } catch (_) {
    return undefined;
  }
};


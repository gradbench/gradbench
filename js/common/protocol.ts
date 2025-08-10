// These are types in the core protocol.

export type Id = number;

export interface Base {
  id: Id;
}

export interface Duration {
  nanoseconds: number;
}

export interface Timing extends Duration {
  name: string;
}

export interface StartMessage extends Base {
  kind: "start";
  eval?: string;
  config?: any;
}

export interface DefineMessage extends Base {
  kind: "define";
  module: string;
}

export interface EvaluateMessage extends Base {
  kind: "evaluate";
  module: string;
  function: string;
  input: any;
  description?: string;
}

export interface AnalysisMessage extends Base {
  kind: "analysis";
  of: Id;
  valid: boolean;
  error?: string;
}

export type Message =
  | StartMessage
  | DefineMessage
  | EvaluateMessage
  | AnalysisMessage;

export interface StartResponse extends Base {
  tool?: string;
  config?: any;
}

export interface DefineResponse extends Base {
  success: boolean;
  timings?: Timing[];
  error?: string;
}

export interface EvaluateResponse extends Base {
  success: boolean;
  output?: any;
  timings?: Timing[];
  error?: string;
}

export type Response = Base | StartResponse | DefineResponse | EvaluateResponse;

export interface Line {
  elapsed: Duration;
}

export interface MessageLine extends Line {
  message: Message;
}

export interface ResponseLine extends Line {
  response: Response;
}

export type Session = (MessageLine | ResponseLine)[];

// These are auxiliary types used by some evals.

/** An integer. */
export type Int = number;

/** A double-precision floating point value. */
export type Float = number;

/**
 * Fields to be included in the input of an eval requesting a tool to run a
 * function multiple times in a single evaluate message. The tool's response
 * should include one timing entry with the name `"evaluate"` for each time it
 * ran the function.
 */
export interface Runs {
  /** Evaluate the function at least this many times. */
  min_runs: number;

  /** Evaluate the function until the total time exceeds this many seconds. */
  min_seconds: number;
}

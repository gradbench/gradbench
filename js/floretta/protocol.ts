type Id = number;

interface Base {
  id: Id;
}

interface Duration {
  nanoseconds: number;
}

export interface Timing extends Duration {
  name: string;
}

export interface Runs {
  min_runs: number;
  min_seconds: number;
}

interface StartMessage extends Base {
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

interface AnalysisMessage extends Base {
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

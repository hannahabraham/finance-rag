export type StepName =
  | "query_understanding"
  | "retrieval"
  | "evidence_verification"
  | "answer_writing"
  | "critic";

export type StepStatus = "pending" | "active" | "complete" | "retry";

export interface ChunkSummary {
  doc_name: string;
  page_number: number | string;
  company?: string;
  score?: number;
  text: string;
}

export interface SourceRef {
  doc_name: string;
  page_number: number | string;
}

export interface StepData {
  // query_understanding
  company?: string;
  doc_period?: string;
  intent?: string;
  parsed_question?: string;
  // retrieval
  chunk_count?: number;
  top_chunks?: ChunkSummary[];
  // evidence_verification
  verified_count?: number;
  kept_chunks?: ChunkSummary[];
  // answer_writing
  answer?: string;
  explanation?: string;
  confidence?: string;
  sources?: SourceRef[];
  // critic
  critique?: string;
  needs_retry?: boolean;
  retry_count?: number;
}

export interface StepEvent {
  step: StepName;
  iteration: number;
  duration_ms: number;
  data: StepData;
}

export interface FinalResult {
  question: string;
  company?: string;
  answer: string;
  explanation?: string;
  confidence: string;
  sources: SourceRef[];
  evidence_snippets: { text: string; doc_name: string; page_number: number | string }[];
  critique?: string;
  iterations?: number;
}

export interface PipelineState {
  status: "idle" | "running" | "done" | "error";
  currentIteration: number;
  steps: StepEvent[];
  result: FinalResult | null;
  error: string | null;
}

export const STEP_ORDER: StepName[] = [
  "query_understanding",
  "retrieval",
  "evidence_verification",
  "answer_writing",
  "critic",
];

export const STEP_LABELS: Record<StepName, string> = {
  query_understanding: "Query Understanding",
  retrieval: "Retrieval",
  evidence_verification: "Evidence Verification",
  answer_writing: "Answer Writing",
  critic: "Critic",
};

export const STEP_DESCRIPTIONS: Record<StepName, string> = {
  query_understanding: "Extracting company, time period, and financial intent from the question",
  retrieval: "Searching the document index for relevant passages",
  evidence_verification: "Filtering retrieved chunks to keep only grounded evidence",
  answer_writing: "Generating a cited answer strictly from verified evidence",
  critic: "Evaluating answer quality and deciding whether to accept or retry",
};

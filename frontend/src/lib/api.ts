import type { StepEvent, FinalResult } from "../types";

export type SSEEvent =
  | { type: "iteration_start"; iteration: number }
  | { type: "step_complete"; step: string; iteration: number; duration_ms: number; data: StepEvent["data"] }
  | { type: "done"; result: FinalResult }
  | { type: "error"; message: string };

export async function streamPipeline(
  question: string,
  mode: string,
  retrieval: string,
  onEvent: (event: SSEEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode, retrieval }),
    signal,
  });

  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.error || `Server error ${resp.status}`);
  }

  const reader = resp.body?.getReader();
  if (!reader) throw new Error("No response stream");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;
      try {
        const event: SSEEvent = JSON.parse(trimmed.slice(6));
        onEvent(event);
      } catch {
        // skip malformed lines
      }
    }
  }
}

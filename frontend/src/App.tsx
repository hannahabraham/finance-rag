import { useCallback, useRef, useState } from "react";
import Header from "./components/Header";
import QuestionForm from "./components/QuestionForm";
import PipelineTracker from "./components/PipelineTracker";
import AnswerCard from "./components/AnswerCard";
import { streamPipeline, type SSEEvent } from "./lib/api";
import type { PipelineState, StepEvent } from "./types";

const INITIAL_STATE: PipelineState = {
  status: "idle",
  currentIteration: 1,
  steps: [],
  result: null,
  error: null,
};

export default function App() {
  const [pipeline, setPipeline] = useState<PipelineState>(INITIAL_STATE);
  const abortRef = useRef<AbortController | null>(null);

  const handleSubmit = useCallback(
    async (question: string, mode: string, retrieval: string) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setPipeline({
        status: "running",
        currentIteration: 1,
        steps: [],
        result: null,
        error: null,
      });

      const handleEvent = (event: SSEEvent) => {
        setPipeline((prev) => {
          switch (event.type) {
            case "iteration_start":
              return { ...prev, currentIteration: event.iteration };

            case "step_complete": {
              const step: StepEvent = {
                step: event.step as StepEvent["step"],
                iteration: event.iteration,
                duration_ms: event.duration_ms,
                data: event.data,
              };
              return { ...prev, steps: [...prev.steps, step] };
            }

            case "done":
              return { ...prev, status: "done", result: event.result };

            case "error":
              return { ...prev, status: "error", error: event.message };

            default:
              return prev;
          }
        });
      };

      try {
        await streamPipeline(
          question,
          mode,
          retrieval,
          handleEvent,
          controller.signal,
        );
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setPipeline((prev) => ({
          ...prev,
          status: "error",
          error: err instanceof Error ? err.message : "Unknown error",
        }));
      }
    },
    [],
  );

  const totalDuration = pipeline.steps.reduce(
    (sum, s) => sum + s.duration_ms,
    0,
  );

  return (
    <div className="min-h-screen bg-slate-50">
      <Header />

      <main className="mx-auto max-w-5xl px-6 py-8">
        <div className="space-y-6">
          {/* Question form */}
          <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
            <QuestionForm
              onSubmit={handleSubmit}
              isLoading={pipeline.status === "running"}
            />
          </section>

          {/* Pipeline progress */}
          {pipeline.status !== "idle" && (
            <section>
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-700">
                  Pipeline Progress
                </h2>
                {pipeline.steps.length > 0 && (
                  <span className="text-xs text-slate-400">
                    Total:{" "}
                    {totalDuration >= 1000
                      ? `${(totalDuration / 1000).toFixed(1)}s`
                      : `${totalDuration}ms`}
                  </span>
                )}
              </div>
              <PipelineTracker pipeline={pipeline} />
            </section>
          )}

          {/* Error */}
          {pipeline.status === "error" && pipeline.error && (
            <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              <p className="font-medium">Pipeline Error</p>
              <p className="mt-1 text-xs">{pipeline.error}</p>
            </div>
          )}

          {/* Final answer */}
          {pipeline.status === "done" && pipeline.result && (
            <section>
              <h2 className="mb-3 text-sm font-semibold text-slate-700">
                Result
              </h2>
              <AnswerCard result={pipeline.result} />
            </section>
          )}
        </div>
      </main>
    </div>
  );
}

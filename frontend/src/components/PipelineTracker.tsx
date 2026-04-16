import {
  BrainCircuit,
  Search,
  ShieldCheck,
  PenLine,
  MessageSquareWarning,
  Check,
  Loader2,
  RotateCcw,
} from "lucide-react";
import type { StepName, PipelineState } from "../types";
import { STEP_ORDER, STEP_LABELS, STEP_DESCRIPTIONS } from "../types";
import StepCard from "./StepCard";

const STEP_ICONS: Record<StepName, React.ReactNode> = {
  query_understanding: <BrainCircuit size={16} />,
  retrieval: <Search size={16} />,
  evidence_verification: <ShieldCheck size={16} />,
  answer_writing: <PenLine size={16} />,
  critic: <MessageSquareWarning size={16} />,
};

interface Props {
  pipeline: PipelineState;
}

export default function PipelineTracker({ pipeline }: Props) {
  if (pipeline.status === "idle") return null;

  const completedSteps = new Set(
    pipeline.steps
      .filter((s) => s.iteration === pipeline.currentIteration)
      .map((s) => s.step),
  );

  const lastCompleted =
    pipeline.steps.length > 0
      ? pipeline.steps[pipeline.steps.length - 1].step
      : null;

  const activeStepIdx = lastCompleted
    ? STEP_ORDER.indexOf(lastCompleted) + 1
    : 0;
  const activeStep =
    pipeline.status === "running" && activeStepIdx < STEP_ORDER.length
      ? STEP_ORDER[activeStepIdx]
      : null;

  // For baseline mode, only show the steps that actually ran
  const relevantSteps =
    pipeline.steps.length > 0
      ? STEP_ORDER.filter(
          (s) =>
            completedSteps.has(s) ||
            s === activeStep ||
            pipeline.steps.some((ev) => ev.step === s),
        )
      : STEP_ORDER;

  // Fall back to full order if relevantSteps is empty (still running first step)
  const stepsToShow =
    relevantSteps.length > 0 ? relevantSteps : STEP_ORDER;

  return (
    <div className="space-y-3">
      {/* Iteration badge */}
      {pipeline.currentIteration > 1 && (
        <div className="flex items-center gap-2 text-xs font-medium text-amber-700">
          <RotateCcw size={14} />
          Iteration {pipeline.currentIteration} — critic requested a retry
        </div>
      )}

      {/* Horizontal stepper */}
      <div className="flex items-center gap-1">
        {stepsToShow.map((stepName, idx) => {
          const isComplete = completedSteps.has(stepName);
          const isActive = stepName === activeStep;
          const stepEvent = pipeline.steps.find(
            (s) =>
              s.step === stepName &&
              s.iteration === pipeline.currentIteration,
          );
          const wasRetry =
            stepEvent?.data?.needs_retry === true;

          return (
            <div key={stepName} className="flex items-center gap-1">
              {idx > 0 && (
                <div
                  className={`h-px w-6 transition-colors ${
                    isComplete ? "bg-emerald-400" : "bg-slate-200"
                  }`}
                />
              )}
              <div
                className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-all ${
                  isComplete && !wasRetry
                    ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                    : isComplete && wasRetry
                      ? "border-amber-200 bg-amber-50 text-amber-700"
                      : isActive
                        ? "border-indigo-300 bg-indigo-50 text-indigo-700"
                        : "border-slate-200 bg-white text-slate-400"
                }`}
              >
                {isComplete && !wasRetry ? (
                  <Check size={13} />
                ) : isActive ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <span className="opacity-60">
                    {STEP_ICONS[stepName]}
                  </span>
                )}
                <span className="hidden sm:inline">
                  {STEP_LABELS[stepName]}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Step detail cards */}
      <div className="space-y-2">
        {stepsToShow.map((stepName) => {
          const stepsForNode = pipeline.steps.filter(
            (s) => s.step === stepName,
          );
          if (stepsForNode.length === 0 && stepName !== activeStep)
            return null;

          const isActive = stepName === activeStep;
          const latestEvent = stepsForNode[stepsForNode.length - 1] ?? null;

          return (
            <StepCard
              key={`${stepName}-${pipeline.currentIteration}`}
              stepName={stepName}
              label={STEP_LABELS[stepName]}
              description={STEP_DESCRIPTIONS[stepName]}
              icon={STEP_ICONS[stepName]}
              isActive={isActive}
              event={latestEvent}
            />
          );
        })}
      </div>
    </div>
  );
}

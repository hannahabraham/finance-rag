import { useState } from "react";
import { ChevronDown, ChevronRight, Clock } from "lucide-react";
import type { StepName, StepEvent } from "../types";

interface Props {
  stepName: StepName;
  label: string;
  description: string;
  icon: React.ReactNode;
  isActive: boolean;
  event: StepEvent | null;
}

export default function StepCard({
  stepName,
  label,
  description,
  icon,
  isActive,
  event,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const data = event?.data;

  return (
    <div
      className={`rounded-lg border transition-all ${
        isActive
          ? "border-indigo-200 bg-indigo-50/50"
          : event
            ? "border-slate-200 bg-white"
            : "border-slate-100 bg-slate-50/50"
      }`}
    >
      <button
        type="button"
        onClick={() => event && setExpanded((v) => !v)}
        className="flex w-full items-center gap-3 px-4 py-3 text-left"
      >
        <span
          className={`flex-shrink-0 ${isActive ? "text-indigo-600 animate-pulse-subtle" : event ? "text-slate-600" : "text-slate-300"}`}
        >
          {icon}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span
              className={`text-sm font-medium ${isActive ? "text-indigo-700" : event ? "text-slate-800" : "text-slate-400"}`}
            >
              {label}
            </span>
            {isActive && (
              <span className="rounded bg-indigo-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-indigo-600">
                Running
              </span>
            )}
            {event && !isActive && (
              <span className="flex items-center gap-1 text-[11px] text-slate-400">
                <Clock size={11} />
                {event.duration_ms >= 1000
                  ? `${(event.duration_ms / 1000).toFixed(1)}s`
                  : `${event.duration_ms}ms`}
              </span>
            )}
          </div>
          <p className="mt-0.5 text-xs text-slate-400">{description}</p>
        </div>
        {event && (
          <span className="flex-shrink-0 text-slate-300">
            {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </span>
        )}
      </button>

      {/* Inline summary (always visible when step is done) */}
      {event && !expanded && <StepSummary stepName={stepName} data={data} />}

      {/* Expanded detail */}
      {event && expanded && (
        <div className="border-t border-slate-100 px-4 py-3">
          <StepDetail stepName={stepName} data={data} />
        </div>
      )}
    </div>
  );
}

function StepSummary({
  stepName,
  data,
}: {
  stepName: StepName;
  data: StepEvent["data"] | undefined;
}) {
  if (!data) return null;

  const Tag = ({ children }: { children: React.ReactNode }) => (
    <span className="inline-block rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
      {children}
    </span>
  );

  return (
    <div className="flex flex-wrap gap-1.5 px-4 pb-3">
      {stepName === "query_understanding" && (
        <>
          {data.company && <Tag>Company: {data.company}</Tag>}
          {data.doc_period && <Tag>Period: {data.doc_period}</Tag>}
          {data.intent && <Tag>Intent: {data.intent}</Tag>}
        </>
      )}
      {stepName === "retrieval" && (
        <Tag>{data.chunk_count ?? 0} chunks retrieved</Tag>
      )}
      {stepName === "evidence_verification" && (
        <Tag>{data.verified_count ?? 0} chunks verified</Tag>
      )}
      {stepName === "answer_writing" && data.confidence && (
        <Tag>Confidence: {data.confidence}</Tag>
      )}
      {stepName === "critic" && (
        <Tag>
          {data.needs_retry ? "Retry requested" : "Accepted"}
        </Tag>
      )}
    </div>
  );
}

function StepDetail({
  stepName,
  data,
}: {
  stepName: StepName;
  data: StepEvent["data"] | undefined;
}) {
  if (!data) return <p className="text-xs text-slate-400">No data</p>;

  if (stepName === "query_understanding") {
    return (
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
        {data.company && (
          <>
            <dt className="font-medium text-slate-500">Company</dt>
            <dd className="text-slate-700">{data.company}</dd>
          </>
        )}
        {data.doc_period && (
          <>
            <dt className="font-medium text-slate-500">Period</dt>
            <dd className="text-slate-700">{data.doc_period}</dd>
          </>
        )}
        {data.intent && (
          <>
            <dt className="font-medium text-slate-500">Intent</dt>
            <dd className="text-slate-700">{data.intent}</dd>
          </>
        )}
        {data.parsed_question && (
          <>
            <dt className="font-medium text-slate-500">Parsed</dt>
            <dd className="text-slate-700">{data.parsed_question}</dd>
          </>
        )}
      </dl>
    );
  }

  if (stepName === "retrieval" || stepName === "evidence_verification") {
    const chunks =
      stepName === "retrieval" ? data.top_chunks : data.kept_chunks;
    return (
      <div className="space-y-2">
        <p className="text-xs font-medium text-slate-500">
          {stepName === "retrieval"
            ? `${data.chunk_count ?? 0} chunks retrieved`
            : `${data.verified_count ?? 0} chunks kept`}
        </p>
        {chunks?.map((c, i) => (
          <div key={i} className="rounded border border-slate-100 p-2">
            <div className="mb-1 flex items-center gap-2 text-[11px] font-medium text-slate-500">
              <span>{c.doc_name}</span>
              <span>p. {c.page_number}</span>
              {c.score != null && (
                <span className="text-slate-400">
                  score {Number(c.score).toFixed(4)}
                </span>
              )}
            </div>
            <p className="text-xs leading-relaxed text-slate-600">{c.text}</p>
          </div>
        ))}
      </div>
    );
  }

  if (stepName === "answer_writing") {
    return (
      <div className="space-y-2 text-xs">
        {data.answer && (
          <p className="leading-relaxed text-slate-700">{data.answer}</p>
        )}
        {data.explanation && (
          <p className="text-slate-500">
            <span className="font-medium">Explanation:</span>{" "}
            {data.explanation}
          </p>
        )}
      </div>
    );
  }

  if (stepName === "critic") {
    return (
      <div className="space-y-1 text-xs">
        <p className="text-slate-700">
          <span className="font-medium">Verdict:</span>{" "}
          {data.needs_retry ? (
            <span className="text-amber-600">Retry</span>
          ) : (
            <span className="text-emerald-600">Accept</span>
          )}
        </p>
        {data.critique && (
          <p className="text-slate-500">{data.critique}</p>
        )}
      </div>
    );
  }

  return (
    <pre className="whitespace-pre-wrap text-xs text-slate-500">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

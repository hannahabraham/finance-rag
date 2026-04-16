import { useState } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  FileText,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import type { FinalResult } from "../types";

const CONFIDENCE_CONFIG: Record<
  string,
  { icon: React.ReactNode; color: string; bg: string; border: string }
> = {
  High: {
    icon: <CheckCircle2 size={16} />,
    color: "text-emerald-700",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
  },
  Medium: {
    icon: <HelpCircle size={16} />,
    color: "text-amber-700",
    bg: "bg-amber-50",
    border: "border-amber-200",
  },
  Low: {
    icon: <AlertTriangle size={16} />,
    color: "text-red-700",
    bg: "bg-red-50",
    border: "border-red-200",
  },
};

export default function AnswerCard({ result }: { result: FinalResult }) {
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const conf = CONFIDENCE_CONFIG[result.confidence] ?? CONFIDENCE_CONFIG.Low;

  return (
    <div className="space-y-4 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      {/* Confidence + sources row */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div
          className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ${conf.color} ${conf.bg} ${conf.border}`}
        >
          {conf.icon}
          {result.confidence} Confidence
        </div>

        {result.sources?.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {result.sources.map((s, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-600"
              >
                <FileText size={12} />
                {s.doc_name}
                {s.page_number != null && `, p.${s.page_number}`}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Answer */}
      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
          Answer
        </h3>
        <p className="text-sm leading-relaxed text-slate-800">
          {result.answer}
        </p>
      </div>

      {/* Explanation */}
      {result.explanation && (
        <div>
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Explanation
          </h3>
          <p className="text-sm leading-relaxed text-slate-600">
            {result.explanation}
          </p>
        </div>
      )}

      {/* Critic notes */}
      {result.critique && (
        <div className="rounded-md bg-slate-50 p-3">
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Critic Notes
          </h3>
          <p className="text-xs text-slate-600">{result.critique}</p>
        </div>
      )}

      {/* Evidence snippets */}
      {result.evidence_snippets?.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setEvidenceOpen((v) => !v)}
            className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400 transition-colors hover:text-slate-600"
          >
            {evidenceOpen ? (
              <ChevronDown size={14} />
            ) : (
              <ChevronRight size={14} />
            )}
            Retrieved Evidence ({result.evidence_snippets.length})
          </button>
          {evidenceOpen && (
            <div className="mt-2 space-y-2">
              {result.evidence_snippets.map((s, i) => (
                <div
                  key={i}
                  className="rounded-md border border-slate-100 bg-slate-50 p-3"
                >
                  <div className="mb-1 text-[11px] font-medium text-slate-500">
                    {s.doc_name}, p.{s.page_number}
                  </div>
                  <p className="text-xs leading-relaxed text-slate-600">
                    {s.text}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

import { useState } from "react";
import { Search, Loader2 } from "lucide-react";

interface Props {
  onSubmit: (question: string, mode: string, retrieval: string) => void;
  isLoading: boolean;
}

const EXAMPLES = [
  "What was Amazon's operating income in 2022?",
  "What risk factors did Adobe disclose in their latest 10-K?",
  "How did 3M describe litigation exposure?",
];

export default function QuestionForm({ onSubmit, isLoading }: Props) {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("multiagent");
  const [retrieval, setRetrieval] = useState("hybrid");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;
    onSubmit(question.trim(), mode, retrieval);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="flex gap-3">
        <div className="relative flex-1">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about public company filings..."
            className="w-full rounded-lg border border-slate-300 bg-white py-3 pl-4 pr-12 text-sm shadow-sm transition-colors placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
          />
        </div>
        <button
          type="submit"
          disabled={isLoading || !question.trim()}
          className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-3 text-sm font-medium text-white shadow-sm transition-colors hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isLoading ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Search size={16} />
          )}
          {isLoading ? "Running..." : "Ask"}
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <fieldset className="flex items-center gap-3">
          <legend className="sr-only">Pipeline Mode</legend>
          <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Mode
          </span>
          {[
            { value: "multiagent", label: "Multi-Agent" },
            { value: "baseline", label: "Baseline" },
          ].map((opt) => (
            <label
              key={opt.value}
              className={`cursor-pointer rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                mode === opt.value
                  ? "border-indigo-600 bg-indigo-50 text-indigo-700"
                  : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
              }`}
            >
              <input
                type="radio"
                name="mode"
                value={opt.value}
                checked={mode === opt.value}
                onChange={() => setMode(opt.value)}
                className="sr-only"
              />
              {opt.label}
            </label>
          ))}
        </fieldset>

        <fieldset className="flex items-center gap-3">
          <legend className="sr-only">Retrieval Strategy</legend>
          <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Retrieval
          </span>
          {[
            { value: "hybrid", label: "Hybrid" },
            { value: "dense", label: "Dense" },
            { value: "bm25", label: "BM25" },
          ].map((opt) => (
            <label
              key={opt.value}
              className={`cursor-pointer rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                retrieval === opt.value
                  ? "border-indigo-600 bg-indigo-50 text-indigo-700"
                  : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
              }`}
            >
              <input
                type="radio"
                name="retrieval"
                value={opt.value}
                checked={retrieval === opt.value}
                onChange={() => setRetrieval(opt.value)}
                className="sr-only"
              />
              {opt.label}
            </label>
          ))}
        </fieldset>
      </div>

      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-slate-400">Try:</span>
        {EXAMPLES.map((ex) => (
          <button
            key={ex}
            type="button"
            onClick={() => setQuestion(ex)}
            className="rounded-md bg-slate-100 px-2.5 py-1 text-xs text-slate-600 transition-colors hover:bg-slate-200"
          >
            {ex}
          </button>
        ))}
      </div>
    </form>
  );
}

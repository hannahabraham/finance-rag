import { BarChart3 } from "lucide-react";

export default function Header() {
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-5xl items-center gap-3 px-6 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 text-white">
          <BarChart3 size={20} />
        </div>
        <div>
          <h1 className="text-lg font-semibold leading-tight text-slate-900">
            Financial Research Assistant
          </h1>
          <p className="text-sm text-slate-500">
            Multi-agent RAG over SEC filing documents
          </p>
        </div>
      </div>
    </header>
  );
}

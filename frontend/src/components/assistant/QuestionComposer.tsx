"use client";

import React, { KeyboardEvent } from "react";
import { Send, Loader2, Search, RotateCcw } from "lucide-react";
import { Document } from "@/lib/types";
import { DocumentSelector } from "./DocumentSelector";
import { TopKControl } from "./TopKControl";

export interface QuestionComposerProps {
  question: string;
  onChangeQuestion: (q: string) => void;
  onSubmit: () => void;
  isLoading: boolean;
  topK: number;
  onChangeTopK: (k: number) => void;
  documents: Document[];
  selectedDocIds: string[];
  onToggleDoc: (docId: string) => void;
  onSelectAllDocs: () => void;
  onClearAllDocs: () => void;
  /** Document list is still loading from the backend. */
  documentsLoading?: boolean;
  /** Document list could not be loaded; distinct from an empty corpus. */
  documentsError?: string | null;
  className?: string;
}

const EXAMPLE_QUESTIONS = [
  "Compare the singular value decomposition and feature weighting approach in GFWSVD with baseline representations.",
  "What objective does LEMS formulate for layer-wise rank selection in LLM compression?",
  "What are the primary methodological differences and evaluation criteria between GFWSVD and LEMS?",
];

export function QuestionComposer({
  question,
  onChangeQuestion,
  onSubmit,
  isLoading,
  topK,
  onChangeTopK,
  documents,
  selectedDocIds,
  onToggleDoc,
  onSelectAllDocs,
  onClearAllDocs,
  documentsLoading = false,
  documentsError = null,
  className = "",
}: QuestionComposerProps) {
  const isQuestionEmpty = !question.trim();

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      if (!isQuestionEmpty && !isLoading) {
        onSubmit();
      }
    }
  };

  return (
    <div className={`space-y-3.5 ${className}`}>
      <DocumentSelector
        documents={documents}
        selectedDocIds={selectedDocIds}
        onToggleDoc={onToggleDoc}
        onSelectAll={onSelectAllDocs}
        onClearAll={onClearAllDocs}
        disabled={isLoading}
        isLoading={documentsLoading}
        loadError={documentsError}
      />

      {/* Query card */}
      <div className="pt-card space-y-4 p-4 sm:p-5">
        <div>
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <label
              htmlFor="research-question-input"
              className="pt-eyebrow flex items-center gap-1.5 text-ink-600"
            >
              <Search className="h-3.5 w-3.5 text-brand-600" />
              <span>Your research question</span>
            </label>
            <span className="hidden text-[11px] text-ink-400 sm:block">
              <kbd className="rounded border border-ink-200 bg-ink-100 px-1 py-0.5 font-mono text-[10px]">
                Ctrl
              </kbd>
              <span className="mx-1">+</span>
              <kbd className="rounded border border-ink-200 bg-ink-100 px-1 py-0.5 font-mono text-[10px]">
                Enter
              </kbd>
              <span className="ml-1.5">to run</span>
            </span>
          </div>

          <textarea
            id="research-question-input"
            rows={3}
            value={question}
            disabled={isLoading}
            onChange={(e) => onChangeQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask something specific about your papers — a method, a result, or a comparison across two of them…"
            aria-label="Natural language scientific research query"
            className="w-full resize-y rounded-lg border border-ink-200 bg-ink-100/40 p-3 text-[14px] leading-relaxed text-ink-900 transition-colors placeholder:text-ink-400 focus:border-brand-400 focus:bg-white disabled:opacity-60"
          />
        </div>

        <div className="flex flex-col gap-4 border-t border-ink-100 pt-3.5 sm:flex-row sm:items-end sm:justify-between">
          <div className="sm:max-w-xs sm:flex-1">
            <TopKControl
              value={topK}
              onChange={onChangeTopK}
              disabled={isLoading}
            />
          </div>

          <div className="flex shrink-0 items-center justify-end gap-2">
            {question && !isLoading && (
              <button
                type="button"
                onClick={() => onChangeQuestion("")}
                className="pt-btn-ghost"
                title="Clear the question"
              >
                <RotateCcw className="h-3.5 w-3.5" />
                <span>Clear</span>
              </button>
            )}

            <button
              type="button"
              disabled={isQuestionEmpty || isLoading}
              onClick={onSubmit}
              aria-label="Submit research inquiry"
              className="pt-btn-primary px-5 py-2.5"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Synthesizing &amp; Verifying...</span>
                </>
              ) : (
                <>
                  <Send className="h-3.5 w-3.5" />
                  <span>Search &amp; verify</span>
                </>
              )}
            </button>
          </div>
        </div>

        <div className="border-t border-ink-100 pt-3">
          <p className="pt-eyebrow mb-2 text-ink-400">Questions this corpus can answer</p>
          <div className="flex flex-wrap gap-1.5">
            {EXAMPLE_QUESTIONS.map((ex, idx) => (
              <button
                key={idx}
                type="button"
                disabled={isLoading}
                onClick={() => onChangeQuestion(ex)}
                className="max-w-full truncate rounded-md border border-ink-200 bg-ink-100/60 px-2.5 py-1 text-left text-[12px] text-ink-600 transition-colors hover:border-brand-200 hover:bg-brand-50 hover:text-brand-800 disabled:opacity-50"
              >
                &ldquo;{ex}&rdquo;
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

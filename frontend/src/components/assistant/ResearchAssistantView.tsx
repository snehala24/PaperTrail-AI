"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Sparkles } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import {
  Document,
  AnswerRequest,
  AnswerResponse,
} from "@/lib/types";
import { generateAnswer, listDocuments, ApiClientError } from "@/lib/api";
import { QuestionComposer } from "./QuestionComposer";
import { AnswerPanel } from "./AnswerPanel";

export interface ResearchAssistantViewProps {
  selectedDocIds: string[];
  onSelectedDocIdsChange: (docIds: string[]) => void;
  /** Called whenever an answer response is received, with the question that produced it. */
  onAnswerGenerated?: (response: AnswerResponse | null, question?: string) => void;
  className?: string;
}

export function ResearchAssistantView({
  selectedDocIds,
  onSelectedDocIdsChange,
  onAnswerGenerated,
  className = "",
}: ResearchAssistantViewProps) {
  const [question, setQuestion] = useState<string>("");
  const [topK, setTopK] = useState<number>(8);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isDocsLoading, setIsDocsLoading] = useState<boolean>(false);
  const [docsError, setDocsError] = useState<string | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [answerResponse, setAnswerResponse] = useState<AnswerResponse | null>(null);

  // Load documents for scoping selector
  useEffect(() => {
    let isMounted = true;
    const loadDocs = async () => {
      setIsDocsLoading(true);
      setDocsError(null);
      try {
        const docs = await listDocuments();
        if (isMounted) setDocuments(docs);
      } catch (err: unknown) {
        // Surface the failure instead of implying the corpus is empty.
        // Scoping is unavailable, but a question can still run corpus-wide.
        if (isMounted) {
          setDocuments([]);
          setDocsError(err instanceof Error ? err.message : "Failed to load documents.");
        }
      } finally {
        if (isMounted) setIsDocsLoading(false);
      }
    };
    loadDocs();
    return () => {
      isMounted = false;
    };
  }, []);

  const handleToggleDoc = useCallback(
    (docId: string) => {
      const next = selectedDocIds.includes(docId)
        ? selectedDocIds.filter((id) => id !== docId)
        : [...selectedDocIds, docId];
      onSelectedDocIdsChange(next);
    },
    [selectedDocIds, onSelectedDocIdsChange]
  );

  const handleSelectAllDocs = useCallback(() => {
    onSelectedDocIdsChange(documents.map((d) => d.doc_id));
  }, [documents, onSelectedDocIdsChange]);

  const handleClearAllDocs = useCallback(() => {
    onSelectedDocIdsChange([]);
  }, [onSelectedDocIdsChange]);

  const handleSubmit = async () => {
    const trimmed = question.trim();
    if (!trimmed || isLoading) return;

    setIsLoading(true);
    setError(null);
    setAnswerResponse(null);

    const req: AnswerRequest = {
      question: trimmed,
      doc_ids: selectedDocIds.length > 0 ? selectedDocIds : undefined,
      top_k: topK,
    };

    try {
      const res = await generateAnswer(req);
      setAnswerResponse(res);
      onAnswerGenerated?.(res, trimmed);
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        setError(err.detail);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected network or server error occurred.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setAnswerResponse(null);
    setError(null);
    onAnswerGenerated?.(null);
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Page Header */}
      <PageHeader
        title="Research Assistant"
        description="Ask a question of your papers. Every claim is checked against verbatim source text before it reaches you — if it cannot be verified, the answer is withheld rather than guessed."
        badge={
          selectedDocIds.length > 0
            ? `${selectedDocIds.length} ${
                selectedDocIds.length === 1 ? "paper" : "papers"
              } selected`
            : "All indexed papers"
        }
      />

      {/* Question Composer */}
      <section aria-label="Research Question Composer">
        <QuestionComposer
          question={question}
          onChangeQuestion={setQuestion}
          onSubmit={handleSubmit}
          isLoading={isLoading}
          topK={topK}
          onChangeTopK={setTopK}
          documents={documents}
          selectedDocIds={selectedDocIds}
          onToggleDoc={handleToggleDoc}
          onSelectAllDocs={handleSelectAllDocs}
          onClearAllDocs={handleClearAllDocs}
          documentsLoading={isDocsLoading}
          documentsError={docsError}
        />
      </section>

      {/* Answer, Claims, Citations, & Abstention Panel */}
      <section aria-label="Grounded Synthesis & Evidence Audit">
        <AnswerPanel
          response={answerResponse}
          isLoading={isLoading}
          error={error}
          onReset={handleReset}
        />
      </section>
    </div>
  );
}

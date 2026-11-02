"use client";

import React, { useState, useEffect, useCallback } from "react";
import { BookOpen } from "lucide-react";
import { PageHeader } from "@/components/ui/PageHeader";
import { Document, DocumentUploadResponse } from "@/lib/types";
import { listDocuments, ApiClientError } from "@/lib/api";
import { invalidateCorpusStats, loadChunkCounts } from "@/lib/corpus";
import { PdfUpload } from "./PdfUpload";
import { DocumentList } from "./DocumentList";
import { DocumentDetailsModal } from "./DocumentDetailsModal";
import { ChunkExplorerModal } from "./ChunkExplorerModal";

export interface ResearchLibraryViewProps {
  selectedDocIds?: string[];
  onSelectedDocIdsChange?: (docIds: string[]) => void;
  className?: string;
}

export function ResearchLibraryView({
  selectedDocIds: externalSelectedDocIds,
  onSelectedDocIdsChange,
  className = "",
}: ResearchLibraryViewProps) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  // Real stored-chunk counts (GET /documents/{id}/chunks); hidden until loaded.
  const [chunkCounts, setChunkCounts] = useState<Record<string, number> | null>(null);

  // Internal selected document IDs if not controlled externally
  const [internalSelectedDocIds, setInternalSelectedDocIds] = useState<string[]>([]);
  const selectedDocIds = externalSelectedDocIds ?? internalSelectedDocIds;

  const updateSelectedDocIds = useCallback(
    (newIds: string[]) => {
      if (onSelectedDocIdsChange) {
        onSelectedDocIdsChange(newIds);
      } else {
        setInternalSelectedDocIds(newIds);
      }
    },
    [onSelectedDocIdsChange]
  );

  // Modals state
  const [selectedDocForDetails, setSelectedDocForDetails] = useState<Document | null>(null);
  const [isDetailsOpen, setIsDetailsOpen] = useState<boolean>(false);

  const [selectedDocForChunks, setSelectedDocForChunks] = useState<Document | null>(null);
  const [isChunksOpen, setIsChunksOpen] = useState<boolean>(false);

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      // Chunk badges come from the real chunk list; omitted until they load.
      try {
        setChunkCounts(await loadChunkCounts(docs));
      } catch {
        setChunkCounts(null);
      }
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        setError(err.detail);
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to load documents from backend.");
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleUploadSuccess = useCallback(
    (res: DocumentUploadResponse) => {
      // Re-fetch documents to update corpus list, page stats and chunk counts.
      invalidateCorpusStats();
      fetchDocuments();
    },
    [fetchDocuments]
  );

  const handleToggleSelect = useCallback(
    (docId: string) => {
      const next = selectedDocIds.includes(docId)
        ? selectedDocIds.filter((id) => id !== docId)
        : [...selectedDocIds, docId];
      updateSelectedDocIds(next);
    },
    [selectedDocIds, updateSelectedDocIds]
  );

  const handleSelectAll = useCallback(() => {
    updateSelectedDocIds(documents.map((d) => d.doc_id));
  }, [documents, updateSelectedDocIds]);

  const handleClearSelection = useCallback(() => {
    updateSelectedDocIds([]);
  }, [updateSelectedDocIds]);

  const handleViewDetails = useCallback((doc: Document) => {
    setSelectedDocForDetails(doc);
    setIsDetailsOpen(true);
  }, []);

  const handleInspectChunks = useCallback((doc: Document) => {
    setSelectedDocForChunks(doc);
    setIsChunksOpen(true);
  }, []);

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <PageHeader
        title="Research Library"
        description="Every paper in the index, how it was parsed, and the exact passages it can be cited from."
        badge={`${documents.length} ${
          documents.length === 1 ? "paper" : "papers"
        } in the index`}
      />

      {/* PDF Upload Section */}
      <section aria-label="PDF Document Ingestion">
        <PdfUpload onUploadSuccess={handleUploadSuccess} />
      </section>

      {/* Document Corpus Section */}
      <section aria-label="Ingested Documents Corpus">
        <DocumentList
          documents={documents}
          isLoading={isLoading}
          error={error}
          selectedDocIds={selectedDocIds}
          onToggleSelect={handleToggleSelect}
          onSelectAll={handleSelectAll}
          onClearSelection={handleClearSelection}
          onRefresh={() => {
            invalidateCorpusStats();
            fetchDocuments();
          }}
          onViewDetails={handleViewDetails}
          onInspectChunks={handleInspectChunks}
          chunkCounts={chunkCounts}
        />
      </section>

      {/* Modals */}
      <DocumentDetailsModal
        document={selectedDocForDetails}
        isOpen={isDetailsOpen}
        onClose={() => setIsDetailsOpen(false)}
        onInspectChunks={handleInspectChunks}
      />

      <ChunkExplorerModal
        document={selectedDocForChunks}
        isOpen={isChunksOpen}
        onClose={() => setIsChunksOpen(false)}
      />
    </div>
  );
}

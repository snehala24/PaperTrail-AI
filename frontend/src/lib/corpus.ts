/**
 * Shared corpus statistics.
 *
 * The backend exposes document metadata (GET /documents) and the full chunk
 * list per document (GET /documents/{doc_id}/chunks), but no chunk-count field.
 * Counts shown anywhere in the UI are therefore derived from those real
 * endpoints — never hardcoded.
 *
 * Results are cached in-module so the sidebar and the library view share one
 * set of requests. A failed load is not cached.
 */

import { useCallback, useEffect, useState } from "react";
import { getDocumentChunks, listDocuments } from "./api";
import { Document } from "./types";

export interface CorpusStats {
  documents: Document[];
  /** doc_id -> number of stored chunks, counted from the real chunk list. */
  chunkCounts: Record<string, number>;
  totalPages: number;
  totalChunks: number;
}

let cached: Promise<CorpusStats> | null = null;
let countsCache: { key: string; promise: Promise<Record<string, number>> } | null = null;

export function invalidateCorpusStats(): void {
  cached = null;
  countsCache = null;
}

/**
 * Count stored chunks for already-loaded documents via GET /documents/{id}/chunks.
 * Use this when a component already holds the document list — it avoids a second
 * GET /documents round trip. Cached per document set, so a newly ingested paper
 * triggers a fresh load automatically.
 */
export function loadChunkCounts(documents: readonly Document[]): Promise<Record<string, number>> {
  const key = documents.map((doc) => doc.doc_id).join(",");
  if (countsCache && countsCache.key === key) return countsCache.promise;

  const pending = Promise.all(
    documents.map(async (doc) => {
      const chunks = await getDocumentChunks(doc.doc_id);
      return [doc.doc_id, chunks.length] as const;
    })
  ).then((entries) => Object.fromEntries(entries));

  countsCache = { key, promise: pending };
  pending.catch(() => {
    if (countsCache?.promise === pending) countsCache = null;
  });
  return pending;
}

export async function loadCorpusStats(force = false): Promise<CorpusStats> {
  if (cached && !force) return cached;

  const pending = (async (): Promise<CorpusStats> => {
    const documents = await listDocuments();
    const chunkCounts = await loadChunkCounts(documents);
    const totalPages = documents.reduce(
      (acc, doc) => acc + (typeof doc.page_count === "number" ? doc.page_count : 0),
      0
    );
    const totalChunks = Object.values(chunkCounts).reduce((acc, n) => acc + n, 0);
    return { documents, chunkCounts, totalPages, totalChunks };
  })();

  cached = pending;
  pending.catch(() => {
    if (cached === pending) cached = null;
  });
  return pending;
}

export interface UseCorpusStatsResult {
  stats: CorpusStats | null;
  isLoading: boolean;
  error: string | null;
  reload: () => void;
}

export function useCorpusStats(): UseCorpusStatsResult {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStats = useCallback(async (force = false) => {
    setIsLoading(true);
    setError(null);
    try {
      const next = await loadCorpusStats(force);
      setStats(next);
    } catch (err: unknown) {
      setStats(null);
      setError(err instanceof Error ? err.message : "Failed to load corpus statistics.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  const reload = useCallback(() => {
    invalidateCorpusStats();
    fetchStats(true);
  }, [fetchStats]);

  return { stats, isLoading, error, reload };
}

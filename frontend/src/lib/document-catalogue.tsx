"use client";

import React, { createContext, useContext, useMemo } from "react";
import { Document } from "./types";
import { documentDisplayTitle } from "./display";

/**
 * Read-only catalogue of ingested papers, provided once by the workspace shell
 * so evidence views can display human-readable paper titles instead of raw
 * `doc_…` identifiers. Falls back to the identifier when the catalogue is
 * still loading or the paper is not in it — it never invents a title.
 */
const EMPTY_CATALOGUE: Record<string, string> = {};

const DocumentCatalogueContext = createContext<Record<string, string>>(
  EMPTY_CATALOGUE
);

export interface DocumentCatalogueProviderProps {
  documents: Document[];
  children: React.ReactNode;
}

export function DocumentCatalogueProvider({
  documents,
  children,
}: DocumentCatalogueProviderProps) {
  const catalogue = useMemo(() => {
    const map: Record<string, string> = {};
    documents.forEach((doc) => {
      map[doc.doc_id] = documentDisplayTitle(doc);
    });
    return map;
  }, [documents]);

  return (
    <DocumentCatalogueContext.Provider value={catalogue}>
      {children}
    </DocumentCatalogueContext.Provider>
  );
}

/** Title lookup keyed by doc_id. Empty while the corpus is unavailable. */
export function useDocumentCatalogue(): Record<string, string> {
  return useContext(DocumentCatalogueContext);
}

/**
 * Resolve a doc_id to its paper title, or null when unknown. Callers must
 * render the identifier itself as the fallback so provenance is never hidden.
 */
export function usePaperTitleResolver(): (
  docId?: string | null
) => string | null {
  const catalogue = useContext(DocumentCatalogueContext);
  return (docId?: string | null) => (docId ? catalogue[docId] ?? null : null);
}

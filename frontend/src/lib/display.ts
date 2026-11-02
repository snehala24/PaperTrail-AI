import { Document } from "./types";

/**
 * Known extraction failures where the title stored with the document is unusable.
 * Each entry is the paper's own title as it appears on page 1 of the indexed PDF,
 * not an invented label.
 *
 * Ingestion now extracts these titles correctly (page-1 line-based title selection,
 * 2026-11-02), so this table only covers records stored *before* that fix. It can be
 * emptied once those records are re-ingested — re-ingestion touches the corpus, so it
 * is done only with explicit approval.
 */
const TITLE_OVERRIDES: Record<string, string> = {
  doc_e9a0d3128767: "LoRA: Low-Rank Adaptation of Large Language Models",
};

/**
 * Some PDFs set titles with letter spacing, so the largest-font heuristic collects
 * single glyphs: "L RA: L -R A L L - M". That is unreadable but still the extracted
 * value, so display falls back to the filename instead of inventing a title.
 */
export function isGarbledTitle(title: string): boolean {
  const tokens = (title || "").split(/\s+/).filter(Boolean);
  if (tokens.length < 3) return false;
  const singleGlyph = tokens.filter((t) => t.length <= 2).length;
  return singleGlyph / tokens.length >= 0.6;
}

export function documentDisplayTitle(doc: Pick<Document, "doc_id" | "title" | "filename">): string {
  const override = TITLE_OVERRIDES[doc.doc_id];
  if (override) return override;
  if (doc.title && !isGarbledTitle(doc.title)) return doc.title;
  return doc.filename || doc.doc_id;
}

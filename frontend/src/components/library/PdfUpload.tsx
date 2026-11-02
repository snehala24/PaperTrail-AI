"use client";

import React, { useState, useRef, DragEvent, ChangeEvent } from "react";
import { UploadCloud, FileText, CheckCircle2, AlertTriangle, AlertCircle, Loader2, X } from "lucide-react";
import { uploadDocument, ApiClientError } from "@/lib/api";
import { DocumentUploadResponse } from "@/lib/types";

export interface PdfUploadProps {
  onUploadSuccess?: (response: DocumentUploadResponse) => void;
  className?: string;
}

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024; // 25 MB

/**
 * Validates whether a file begins with %PDF- magic bytes.
 */
export async function validatePdfMagicBytes(file: File): Promise<boolean> {
  try {
    const slice = file.slice(0, 5);
    let uint8: Uint8Array;

    if (typeof slice.arrayBuffer === "function") {
      const buffer = await slice.arrayBuffer();
      uint8 = new Uint8Array(buffer);
    } else if (typeof FileReader !== "undefined") {
      const buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          if (reader.result instanceof ArrayBuffer) {
            resolve(reader.result);
          } else {
            const enc = new TextEncoder();
            resolve(enc.encode(String(reader.result)).buffer);
          }
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsArrayBuffer(slice);
      });
      uint8 = new Uint8Array(buffer);
    } else {
      return true;
    }

    if (uint8.length < 5) return false;
    // %PDF- is 0x25, 0x50, 0x44, 0x46, 0x2D
    return (
      uint8[0] === 0x25 &&
      uint8[1] === 0x50 &&
      uint8[2] === 0x44 &&
      uint8[3] === 0x46 &&
      uint8[4] === 0x2d
    );
  } catch {
    return false;
  }
}

export function PdfUpload({ onUploadSuccess, className = "" }: PdfUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const resetFileInput = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const resetState = () => {
    resetFileInput();
    setErrorMessage(null);
  };

  const validateAndSelectFile = async (file: File) => {
    setErrorMessage(null);
    setSuccessMessage(null);

    // 1. Extension and MIME check
    const isPdfExt = file.name.toLowerCase().endsWith(".pdf");
    const isPdfMime = file.type === "application/pdf";
    if (!isPdfExt && !isPdfMime) {
      setErrorMessage(`Only PDF files are supported. Received: "${file.name}"`);
      resetFileInput();
      return;
    }

    // 2. Empty file check
    if (file.size === 0) {
      setErrorMessage("Empty file provided. Please select a valid, non-empty PDF document.");
      resetFileInput();
      return;
    }

    // 3. File size check (25 MB)
    if (file.size > MAX_UPLOAD_BYTES) {
      const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
      setErrorMessage(`File exceeds maximum upload size of 25 MB. Selected file is ${sizeMb} MB.`);
      resetFileInput();
      return;
    }

    // 4. Client-side magic bytes validation (%PDF-)
    const isValidHeader = await validatePdfMagicBytes(file);
    if (!isValidHeader) {
      setErrorMessage(`Invalid PDF format: "${file.name}" lacks valid %PDF- magic byte header.`);
      resetFileInput();
      return;
    }

    setSelectedFile(file);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isUploading) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (isUploading) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      await validateAndSelectFile(droppedFile);
    }
  };

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await validateAndSelectFile(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || isUploading) return;

    setIsUploading(true);
    setErrorMessage(null);
    setSuccessMessage(null);

    try {
      const response = await uploadDocument(selectedFile);
      const title = response.document.title || response.document.filename;
      setSuccessMessage(
        `Successfully ingested "${title}" (${response.document.page_count} pages, ${response.chunk_count} chunks indexed).`
      );
      resetState();
      if (onUploadSuccess) {
        onUploadSuccess(response);
      }
    } catch (err: unknown) {
      if (err instanceof ApiClientError) {
        setErrorMessage(err.detail || `Upload failed with status ${err.status}`);
      } else if (err instanceof Error) {
        setErrorMessage(`Upload failed: ${err.message}`);
      } else {
        setErrorMessage("An unexpected error occurred during paper ingestion.");
      }
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className={`bg-white rounded-xl border border-ink-200/90 p-5 shadow-sm ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-ink-700 flex items-center gap-2">
          <UploadCloud className="w-4 h-4 text-brand-600" />
          Ingest Research Paper
        </h2>
        <span className="text-xs text-ink-500 font-medium">
          PDF up to 25 MB • parsed into citable passages
        </span>
      </div>

      {/* Drag & Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && fileInputRef.current?.click()}
        role="region"
        aria-label="PDF upload dropzone"
        className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-all cursor-pointer select-none ${
          isDragging
            ? "border-brand-500 bg-brand-50/60 ring-2 ring-brand-300 ring-offset-1"
            : isUploading
            ? "border-ink-200 bg-ink-50/80 cursor-not-allowed opacity-80"
            : "border-ink-300 hover:border-brand-400 bg-ink-50/50 hover:bg-ink-50"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          disabled={isUploading}
          onChange={handleFileChange}
          className="sr-only"
          aria-label="Upload PDF file input"
        />

        <div className="flex flex-col items-center justify-center space-y-2">
          {isUploading ? (
            <div className="flex flex-col items-center space-y-2 text-ink-700">
              <Loader2 className="w-8 h-8 text-brand-600 animate-spin" />
              <p className="text-sm font-medium">Ingesting and indexing paper...</p>
              <p className="text-xs text-ink-500 max-w-sm">
                Parsing layout, segmenting chunks, computing SHA-256 deduplication, and updating hybrid retrieval indices.
              </p>
            </div>
          ) : selectedFile ? (
            <div className="flex flex-col items-center space-y-2">
              <div className="w-10 h-10 rounded-full bg-brand-100 flex items-center justify-center text-brand-700">
                <FileText className="w-5 h-5" />
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold text-ink-900">{selectedFile.name}</p>
                <p className="text-xs text-ink-500">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready for ingestion
                </p>
              </div>
            </div>
          ) : (
            <>
              <div className="w-10 h-10 rounded-full bg-ink-100 flex items-center justify-center text-ink-500">
                <UploadCloud className="w-5 h-5" />
              </div>
              <div>
                <p className="text-sm font-medium text-ink-800">
                  <span className="text-brand-600 font-semibold hover:underline">Click to browse</span> or drag and drop paper
                </p>
                <p className="text-xs text-ink-500 mt-0.5">
                  Scientific papers in PDF format with valid %PDF- magic bytes header
                </p>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Selected file action buttons */}
      {selectedFile && !isUploading && (
        <div className="mt-3 flex items-center justify-end space-x-2">
          <button
            type="button"
            onClick={resetState}
            className="px-3 py-1.5 text-xs font-medium text-ink-600 hover:text-ink-900 hover:bg-ink-100 rounded-md transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleUpload}
            className="px-4 py-1.5 text-xs font-semibold text-white bg-brand-600 hover:bg-brand-700 rounded-md shadow-sm transition-colors flex items-center gap-1.5"
          >
            <UploadCloud className="w-3.5 h-3.5" />
            Upload & Ingest Paper
          </button>
        </div>
      )}

      {/* Error alert */}
      {errorMessage && (
        <div
          role="alert"
          className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start justify-between text-xs text-red-700"
        >
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Ingestion Failed</p>
              <p className="mt-0.5">{errorMessage}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setErrorMessage(null)}
            className="text-red-500 hover:text-red-800 p-0.5 rounded transition-colors"
            aria-label="Dismiss error"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Success alert */}
      {successMessage && (
        <div
          role="status"
          className="mt-3 p-3 bg-emerald-50 border border-emerald-200 rounded-lg flex items-start justify-between text-xs text-emerald-800"
        >
          <div className="flex items-start space-x-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Ingestion Completed</p>
              <p className="mt-0.5">{successMessage}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setSuccessMessage(null)}
            className="text-emerald-600 hover:text-emerald-900 p-0.5 rounded transition-colors"
            aria-label="Dismiss success message"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}

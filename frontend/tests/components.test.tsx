import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { StatusIndicator } from "../src/components/ui/StatusIndicator";
import { PageHeader } from "../src/components/ui/PageHeader";
import { EmptyState } from "../src/components/ui/EmptyState";
import { AppSidebar } from "../src/components/layout/AppSidebar";
import { AppHeader } from "../src/components/layout/AppHeader";
import { CorpusStats } from "../src/lib/corpus";
import { BookOpen } from "lucide-react";

describe("Design System & Layout Components (Phase 2)", () => {
  describe("StatusIndicator", () => {
    it("renders Online state with version", () => {
      render(<StatusIndicator status="connected" version="0.1.0" />);
      expect(screen.getByText("Backend v0.1.0 Online")).toBeInTheDocument();
    });

    it("renders Offline state", () => {
      render(<StatusIndicator status="disconnected" />);
      expect(screen.getByText("Backend Offline (:8000)")).toBeInTheDocument();
    });

    it("renders Connecting state", () => {
      render(<StatusIndicator status="checking" />);
      expect(screen.getByText("Connecting...")).toBeInTheDocument();
    });

    it("calls onRefresh when refresh button clicked", () => {
      const handleRefresh = vi.fn();
      render(<StatusIndicator status="connected" onRefresh={handleRefresh} />);
      const btn = screen.getByRole("button", { name: /Refresh backend connection/i });
      fireEvent.click(btn);
      expect(handleRefresh).toHaveBeenCalledTimes(1);
    });
  });

  describe("PageHeader", () => {
    it("renders title, description, and optional badge", () => {
      render(
        <PageHeader
          title="Custom Title"
          description="Custom description text"
          badge="Beta"
          actions={<button>Action</button>}
        />
      );
      expect(screen.getByText("Custom Title")).toBeInTheDocument();
      expect(screen.getByText("Custom description text")).toBeInTheDocument();
      expect(screen.getByText("Beta")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Action" })).toBeInTheDocument();
    });
  });

  describe("EmptyState", () => {
    it("renders icon, title, description, hint, and feature list", () => {
      render(
        <EmptyState
          icon={BookOpen}
          title="Empty Title"
          description="Empty description"
          hint="Nothing indexed yet"
          features={["Feature Alpha", "Feature Beta"]}
        />
      );
      expect(screen.getByText("Empty Title")).toBeInTheDocument();
      expect(screen.getByText("Empty description")).toBeInTheDocument();
      expect(screen.getByText("Nothing indexed yet")).toBeInTheDocument();
      expect(screen.getByText("Feature Alpha")).toBeInTheDocument();
      expect(screen.getByText("Feature Beta")).toBeInTheDocument();
    });
  });

  describe("AppSidebar", () => {
    it("renders PaperTrail AI branding and all 5 navigation items", () => {
      const handleSelect = vi.fn();
      const handleClose = vi.fn();

      render(
        <AppSidebar
          activeSection="assistant"
          onSelectSection={handleSelect}
          isMobileOpen={false}
          onCloseMobile={handleClose}
        />
      );

      expect(screen.getByText("PaperTrail AI")).toBeInTheDocument();
      expect(screen.getByText("Dashboard")).toBeInTheDocument();
      expect(screen.getByText("Research Library")).toBeInTheDocument();
      expect(screen.getByText("Research Assistant")).toBeInTheDocument();
      expect(screen.getByText("Evidence Explorer")).toBeInTheDocument();
      expect(screen.getByText("Cross-Paper Analysis")).toBeInTheDocument();

      // Corpus overview is data-driven: while the backend has not answered, no
      // paper statistics may be presented as fact.
      expect(screen.getByText("Loading…")).toBeInTheDocument();
      expect(screen.queryByText("2 papers")).not.toBeInTheDocument();
      expect(screen.queryByText("53p / 209c")).not.toBeInTheDocument();
      expect(screen.queryByText("LEMS (arXiv:2610)")).not.toBeInTheDocument();
    });

    it("renders ingested corpus statistics from real backend data", () => {
      const corpus: CorpusStats = {
        documents: [
          {
            doc_id: "doc_c67573ef50cc",
            title: "Advancing SVD-based LLM Compression via Layer-Wise Error Model Search",
            filename: "2610.32878v1.pdf",
            sha256: "c67573ef50cc",
            page_count: 34,
            extraction_warnings: [],
          },
          {
            doc_id: "doc_7fdb1200e823",
            title: "Generalized Fisher-Weighted SVD",
            filename: "2505.17974.pdf",
            sha256: "7fdb1200e823",
            page_count: 19,
            extraction_warnings: [],
          },
        ],
        chunkCounts: { doc_c67573ef50cc: 131, doc_7fdb1200e823: 78 },
        totalPages: 53,
        totalChunks: 209,
      };

      render(
        <AppSidebar
          activeSection="library"
          onSelectSection={vi.fn()}
          isMobileOpen={false}
          onCloseMobile={vi.fn()}
          corpus={corpus}
        />
      );

      expect(screen.getByText("2 papers")).toBeInTheDocument();
      expect(screen.getByText("arXiv:2610.32878v1")).toBeInTheDocument();
      expect(screen.getByText("34p • 131c")).toBeInTheDocument();
      expect(screen.getByText("19p • 78c")).toBeInTheDocument();
      expect(screen.getByText("53p / 209c")).toBeInTheDocument();
    });

    it("reports corpus unavailability instead of inventing statistics", () => {
      render(
        <AppSidebar
          activeSection="library"
          onSelectSection={vi.fn()}
          isMobileOpen={false}
          onCloseMobile={vi.fn()}
          corpus={null}
          corpusError="Network error: Failed to fetch"
        />
      );

      expect(screen.getByText("Unavailable")).toBeInTheDocument();
      expect(screen.getByText(/Corpus stats unavailable/)).toBeInTheDocument();
      expect(screen.getByText("—")).toBeInTheDocument();
    });

    it("identifies active navigation item with aria-current", () => {
      render(
        <AppSidebar
          activeSection="library"
          onSelectSection={vi.fn()}
          isMobileOpen={false}
          onCloseMobile={vi.fn()}
        />
      );

      const libraryBtn = screen.getByRole("button", { name: /Research Library/i });
      expect(libraryBtn).toHaveAttribute("aria-current", "page");

      const assistantBtn = screen.getByRole("button", { name: /Research Assistant/i });
      expect(assistantBtn).not.toHaveAttribute("aria-current");
    });

    it("triggers onSelectSection when an item is clicked", () => {
      const handleSelect = vi.fn();
      render(
        <AppSidebar
          activeSection="assistant"
          onSelectSection={handleSelect}
          isMobileOpen={false}
          onCloseMobile={vi.fn()}
        />
      );

      const evidenceBtn = screen.getByRole("button", { name: /Evidence Explorer/i });
      fireEvent.click(evidenceBtn);
      expect(handleSelect).toHaveBeenCalledWith("evidence");
    });
  });

  describe("AppHeader", () => {
    it("renders section title and status indicator", () => {
      const handleOpenMenu = vi.fn();
      const handleRefresh = vi.fn();

      render(
        <AppHeader
          sectionTitle="Grounded Research Assistant"
          sectionSubtitle="Query evidence across papers"
          backendStatus="connected"
          backendVersion="0.1.0"
          onRefreshHealth={handleRefresh}
          onOpenMobileMenu={handleOpenMenu}
        />
      );

      expect(screen.getByText("Grounded Research Assistant")).toBeInTheDocument();
      expect(screen.getByText("Backend v0.1.0 Online")).toBeInTheDocument();

      const menuBtn = screen.getByRole("button", { name: /Open navigation menu/i });
      fireEvent.click(menuBtn);
      expect(handleOpenMenu).toHaveBeenCalledTimes(1);
    });
  });
});

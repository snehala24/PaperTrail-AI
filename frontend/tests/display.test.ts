import { describe, it, expect } from "vitest";
import { documentDisplayTitle, isGarbledTitle } from "../src/lib/display";

describe("documentDisplayTitle", () => {
  it("uses a clean stored title unchanged", () => {
    expect(
      documentDisplayTitle({
        doc_id: "doc_7fdb1200e823",
        title: "Generalized Fisher-Weighted SVD",
        filename: "2505.17974.pdf",
      })
    ).toBe("Generalized Fisher-Weighted SVD");
  });

  it("substitutes the recorded correction for a known extraction failure", () => {
    expect(
      documentDisplayTitle({
        doc_id: "doc_e9a0d3128767",
        title: "L RA: L -R A L L - M",
        filename: "2106.09685v2.pdf",
      })
    ).toBe("LoRA: Low-Rank Adaptation of Large Language Models");
  });

  it("falls back to the filename when a letter-spaced title is unreadable", () => {
    expect(
      documentDisplayTitle({
        doc_id: "doc_unknown00000",
        title: "A B: C D E F G H",
        filename: "9999.00000v1.pdf",
      })
    ).toBe("9999.00000v1.pdf");
  });

  it("never returns an empty string", () => {
    expect(documentDisplayTitle({ doc_id: "doc_aaaaaaaaaaaa", title: "", filename: "" })).toBe(
      "doc_aaaaaaaaaaaa"
    );
  });

  it("recognises sparse-glyph titles but not ordinary short ones", () => {
    expect(isGarbledTitle("L RA: L -R A L L - M")).toBe(true);
    expect(isGarbledTitle("Attention Is All You Need")).toBe(false);
    expect(isGarbledTitle("BERT")).toBe(false);
    expect(isGarbledTitle("")).toBe(false);
  });
});

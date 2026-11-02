# Human review guide — benchmark gold annotations

Protocol version: `1.1` (the version in `backend/app/eval/review_worksheet.py::REVIEW_PROTOCOL_VERSION`)
Written: 2026-11-02 · Companion file: [review_worksheet.md](review_worksheet.md)

**Current state of the dataset this guide covers: 30 questions, 34 gold annotations, 0 human-verified.**
24 annotations are `human_draft` (researcher-drafted, `labelled_by: snehala24`) and 10 are `ai_draft`
(`labelled_by: assistant`). Nothing has been signed off yet, and this guide does not sign anything off —
it exists so that the person who does can do it correctly and can say afterwards what they committed to.

Read this before the worksheet, not while filling it in.

## 1. What you are deciding, and what you are not

You are deciding one thing: **does this recorded annotation faithfully describe the paper as published?**

You are *not* scoring the product. Whether the retrieval system finds the span, whether the answer model
cites it, and whether the conflict detector labels it correctly are all measured separately, against these
labels, only after the labels are trustworthy. Marking an annotation "verified" because the pipeline happens
to retrieve it is the mistake this whole document exists to prevent.

Three further limits worth stating up front:

- The corpus holds five papers, but **all 30 questions target one paper** — LEMS (`doc_c67573ef50cc`,
  `data/2610.32878v1.pdf`) — and only Q-029 and Q-030 also involve GFWSVD (`doc_7fdb1200e823`,
  `data/2505.17974.pdf`). Attention, LoRA and DoRA have no gold annotations at all. A pass rate here is a
  pass rate on one paper, not on five.
- The stored chunk text is machine-extracted with PyMuPDF. It drops figure content and can reflow tables.
  The worksheet's corpus check compares your span against **that extracted text**, never against the PDF.
- 32 of the 34 spans sit in a 34-page paper with 131 chunks; the two remaining ones are in a 19-page paper
  with 78 chunks. Page references are worth checking twice in the first of those.

Open the PDF yourself. That is the only step that can promote a label.

## 2. The five evidence categories

Every judgement you make about a relation falls into one of these. The pipeline is required to keep them
distinct, and the offline tests in `backend/tests/test_relation_taxonomy.py` use explicitly synthetic
fixtures to prove it does — those fixtures are not real research findings and must never be cited as such.

| Category | What it means | What the ledger should record |
| --- | --- | --- |
| Genuine contradiction | Two statements assert opposite values for the same property under comparable conditions; both cannot be true | `contradicts`, with a reason naming the property and the conditions |
| Apparent conflict | The results differ, but a methods, data, metric or setting difference explains the gap | `apparent_conflict`, with that difference named in `conflict_reason` |
| One-sided evidence | Only one side of the pair is grounded in a verified claim | a note, **no** comparison object |
| Insufficient evidence | Neither side is grounded, or retrieval returned nothing usable | abstention — no answer text, no evidence links, no citations |
| Compatible findings | The statements agree, or address different questions entirely | `supports` (or nothing at all) |

The rule that governs the hard cases: **a methods, data, metric or setting difference is never a
contradiction.** Two papers reporting different numbers is the normal condition of research, not a finding.

### The schema caveat you have to know about

`GoldEvidence.expected_relation` is an `EvidenceRelation` enum whose members are
`supports`, `weakens`, `contradicts`, `contextualizes`, `insufficient`. **There is no
`apparent_conflict` member.** An apparent conflict is therefore recorded as `contradicts` (or `weakens`)
plus free text in `conflict_reason`. So when you review one of the two `contradicts` items here, you are
judging the label *and* the reason together, and they can disagree — which, in this dataset, they do. See
section 6.

## 3. Check one: is the evidence exactly what the paper says?

`review_worksheet.md` prints a `- Corpus check:` line for every item, with one of six verdicts:

| Verdict | Meaning | What to do |
| --- | --- | --- |
| `exact` | the span matches the extracted chunk text character for character | still confirm against the PDF — extraction is not the paper |
| `case only` | only letter case differs from the source | decide whether the annotation should read as printed; a `human_verified` label on a span the paper does not print is a defect you just introduced |
| `hyphen or space repair` | it matches only after line-break hyphens are rejoined | check the PDF really breaks the word there; if the annotation glues words the paper keeps hyphenated, say so |
| `other page` | the wording exists, on a different page | the page reference is wrong even though the quote is right |
| `absent` | not in this document's stored text at all | page, wording, or the document itself is wrong — this is the most serious flag |
| `no document` | the recorded `doc_sha256` matches nothing in `data/` | the PDF is missing; re-upload is needed before review, which requires a separate approval |

Today the tally is **33 `exact`, 1 `case only`, 0 `absent`, 0 `other page`, 0 `no document`** — all 34 spans
are structurally present somewhere in the corpus. Structural integrity is not human validation; it says the
annotations were drafted against real text, nothing more.

What the automated check cannot do, and only you can:

- Notice that the span is a substring of a *different* sentence that happens to contain the same words.
- Notice the extraction mangled a superscript, a math symbol, or a table cell (e.g. `2.01×` stored as `2.01x`).
- Notice the span is technically verbatim but truncated mid-clause so it no longer asserts anything.

Two items — GOLD-003 (`5.07 GB`, 7 characters) and GOLD-004 (`60.83`, 5 characters), both on page 24 of the
same paper, both answering Q-003 about Table 9 — are short enough that a coincidental match elsewhere on the
page is a real risk. For table cells, verify that `annotator_notes` names the row and column; a bare number
without its row/column context cannot be re-checked by anyone later, including you.

## 4. Check two: is the page reference right?

`page` is 1-indexed **by PDF page**, not by the printed page number in the paper's footer. Common sources of
a one-off:

- A title page or an arXiv cover sheet that shifts the printed numbering.
- A paper whose page 1 of the PDF is a two-column header continuation.
- A table or figure that spills across a page boundary, where the extractor attached the caption to one
  page and the cells to the next.

If the span is on PDF page 24 and the annotation says 23, that is a defect even if the quote is perfect —
the citation layer resolves against the recorded page, and a reviewer who "knows what they meant" is exactly
the person who produces an unverifiable dataset. Record the correct page in `observations`; do not edit
`page` in the JSONL while you are reviewing (see section 8).

## 5. Check three: does the span actually support the claim?

"Supports" means **entails**, not "is about". Test it this way: read the question, then read the span alone,
with no other sentence. If a careful reader could not answer the question from the span, it does not support
it — even if it came from the right paragraph.

Typical failures to look for:

- The span states a method rather than the reported value the question asks for.
- The span gives the number for a different model, different compression ratio, or different hardware than
  the question specifies (Q-001 asks about Qwen3-8B on an A100 at 80% compression; a 2.01× figure from any
  other setting is not an answer).
- The span is the *setup* ("we evaluate on X") where the answer is a *result*.
- Comparative questions (Q-003, Q-016, Q-020, Q-022) need one span per side of the comparison. If only one
  side is annotated, the gold set is incomplete, and an incomplete gold set silently deflates recall.
- Nine questions carry two gold spans each; for those, both spans must be individually sufficient, and both
  get the same verification decision.

For the five `unanswerable` probes — Q-009, Q-010, Q-026, Q-027, Q-028 — see section 7.

## 6. Check four: the contradiction labels

Two annotations carry `contradicts`: **GOLD-006** (Q-005, p.23) and **GOLD-034** (Q-030, p.9). These are the
hardest calls in the set and the worksheet flags each of them.

Apply this four-part test. All four must hold for `contradicts` to be the right label:

1. Both statements are about the **same property** (same metric, same unit).
2. Both are measured under **comparable conditions** (same model, same dataset, same budget, same hardware).
3. They assert **opposite values** — not merely different ones.
4. Both cannot be true at once.

If test 2 fails but 1, 3 and 4 hold, you have an **apparent conflict**, and the correct ledger relation is
`apparent_conflict` with the differing condition named. If you cannot name the differing condition, the
honest label is `unresolved`, not `contradicts`.

Now the finding you need to make a decision about, stated as the data shows it. Both of these items record
a `conflict_reason` that begins by describing an apparent conflict:

- GOLD-006: *"Apparent conflict: BitStack isolates binary sign masks that cannot be compressed with standard
  quantization kernels, while traditional SVD allows 4-bit quantization across all components."*
- GOLD-034: *"Apparent conflict: LEMS reports GFWSVD degrades significantly under identical conditions,
  whereas GFWSVD reported strong performance in its own evaluation."*

The label says `contradicts`; the annotator's own reason says "apparent conflict". Those point at different
relations, so **at least one of the two fields is wrong**, and only a human reading the PDF can say which.
GOLD-034 is the sharper case: its reason says "under identical conditions", which is precisely the condition
test 2 requires — so it *might* be a genuine contradiction, but only if the two reported numbers are for the
same model, same compression ratio and same metric. Verify that against both papers, and write down what you
compared. If the conditions differ, the label should become `apparent_conflict` and the pipeline should be
expected to abstain from `contradicts`.

This is also why the offline conflict benchmark currently reports **0/4 relation-label matches and 0
unearned `contradicts` emissions** (`eval/runs/run_20261102_150550_conflicts.json`): no verified
contradiction exists in the corpus to detect, and the system is right to decline to invent one. Your review
does not fix that; a validated label at least tells us whether one of these two is a real case.

## 7. Check five: abstention expectations

Five questions are `unanswerable` probes. They carry **no gold spans** — that absence is deliberate: the
correct behaviour is the system saying it cannot ground an answer, so a gold span would be a contradiction in
terms. The worksheet prints a short block for each. To review one:

1. Search the paper for the requested fact (the question text names it).
2. Confirm the corpus genuinely lacks it — not that it is merely hard to find, but that the paper does not
   report it.
3. Write one line in `observations` saying what you searched and what you found instead.

Two failure modes to report explicitly rather than tick past:

- **Accidentally answerable.** If the paper *does* contain the fact, the probe is mis-specified and the
  abstention metric is measuring the wrong thing. Say so; the fix is to re-specify the question, which is a
  dataset change that needs its own approval.
- **Answerable only by inference.** If the fact can be derived by combining two tables but is never stated,
  record that the expected behaviour is abstention-with-no-citation, not a derived answer. The citation layer
  validates verbatim spans, so a derived answer cannot be grounded and must not be counted as a success.

For the five `apparent_conflict`-typed questions (Q-005, Q-006, Q-015, Q-017, Q-030), note that most of
their gold spans are labelled `supports`. That is often correct — a conflict question needs one grounded span
per side, and each side by itself is a supporting statement. The worksheet marks those items with a caution
so you say which reading applies instead of guessing. The expected ledger relation for the *pair* is
`apparent_conflict`, which no `expected_relation` field can express today; the clearest thing you can do for
those items is confirm both spans are verbatim and sufficient, and state in `observations` what the differing
condition is.

## 8. Recording a verdict

For each item, tick at most one of the checkboxes in its block and fill the fenced block:

```text
reviewer handle : human:<your-handle>
review date     : YYYY-MM-DD
observations    : what you opened, what you compared, what you concluded
```

Then, in `eval/gold_evidence.jsonl`, for the items you actually accepted:

- `status`: `human_draft` / `ai_draft` → `human_verified`
- `labelled_by`: → `human:<your-handle>`
- `annotator_notes`: append what you verified, in your own words

Three hard rules.

**Do not promote a label without a handle.** `--check` fails on any item marked `human_verified` whose
`labelled_by` does not start with `human:` — that guard is a test, not a convention
(`backend/tests/test_eval_review_worksheet.py::TestLabelIntegrity`).

**Do not edit `answer_span`, `page`, `doc_sha256`, `question`, or `expected_relation` while reviewing.**
Those fields define the benchmark; changing them changes what past runs measured, and `eval/runs/` artifacts
would then describe a dataset that no longer exists. If a span or page is wrong, record it in
`observations` and treat the correction as a separate, explicitly approved dataset revision, re-run the
benchmarks, and keep the old artifacts with a note that the gold labels changed.

**Leave anything you did not personally read in the PDF as a draft.** Half-reviewed is worse than unreviewed:
the dataset then contains labels that look promoted. The provenance fields exist so that a reader can tell
draft from verified — do not blur them.

Run these after editing:

```bash
.venv/Scripts/python.exe -m backend.app.eval.review_worksheet --check   # label integrity only
.venv/Scripts/python.exe -m backend.app.eval.review_worksheet           # regenerate the worksheet
```

`--check` reads the JSONL and never writes it. The generator likewise only writes
`review_worksheet.md`. Neither can promote a status.

## 9. A worked example

GOLD-010 (Q-008, page 5, LEMS) records the span `Integer Linear Program`. The corpus check says **case
only**: the stored text on that page reads `Integer linear program (ILP)` — lowercase except the initial
letter. So:

1. Open `data/2610.32878v1.pdf` at PDF page 5 and read what the paper actually prints.
2. If the paper prints it title-case inside a section heading, the annotation is verbatim and the mismatch is
   an artefact of the extractor's body-text chunk — verify the label, and note in `observations` that page 5
   heading capitalisation was the reason for the flag.
3. If the paper prints it lowercase, the span should read as printed. Note the correction; do not edit the
   JSONL mid-review.
4. Either way, also confirm the span answers Q-008 — a term appearing on a page is not evidence for a claim
   about it.

That is the whole method: read the paper, say what you saw, and let the fields reflect it.

## 10. Suggested order of work

1. The **15 flagged items** listed under "Where to start" in the worksheet. They carry 20 cautions between
   them: 2 `contradicts` labels (each also flagged for the label/reason disagreement in section 6), 4
   cross-paper items, 7 conflict-typed questions whose span is labelled `supports`, 2 short spans,
   1 case-only span, and 2 duplicate-span notes. The counts overlap because one item can raise several
   cautions. GOLD-025 (Q-021) and GOLD-033 (Q-030) record the identical page-6 span, so reading it once
   decides both blocks — record the same verdict in each.
2. The two cross-paper questions Q-029 and Q-030, because they need both PDFs open side by side.
3. The remaining 19 items, spot-checking at least five against the full surrounding paragraph rather than the
   quoted fragment alone.

Rough budget: the 15 flagged items take about 5 minutes each with two PDFs open; the rest about 2 minutes.
That is roughly 1.5–2 hours for the whole set — which is the real cost of claiming human-verified ground
truth, and the reason no accuracy number is published until it is paid.

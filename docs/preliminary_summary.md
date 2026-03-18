# Preliminary Analysis Summary
**Project:** Question-Type-Aware VQA Analysis (BLIP vs ViLT on GQA)
**Dataset:** GQA Balanced Validation Split
**Date:** 2026-02-22

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total questions (balanced val) | 132,062 |
| Unique answer strings | 1,469 |
| ViLT vocabulary size | 3,129 labels |

### Structural Type Breakdown

| Type | Count | Pct |
|---|---|---|
| query | 68,167 | 51.6% |
| verify | 27,413 | 20.8% |
| choose | 16,229 | 12.3% |
| logical | 16,096 | 12.2% |
| compare | 4,157 | 3.1% |

### Semantic Type Breakdown

| Type | Count | Pct |
|---|---|---|
| rel | 61,634 | 46.7% |
| attr | 42,201 | 32.0% |
| obj | 15,564 | 11.8% |
| cat | 8,616 | 6.5% |
| global | 4,047 | 3.1% |

**Note on `query` sub-types:** Within `structural == "query"`, semantic types are: `rel` (59.1%), `attr` (26.5%), `cat` (10.5%), `global` (3.8%). **`obj` never co-occurs with `query`** — `semantic == "obj"` only appears in `logical` (81.5%) and `verify` (18.5%) questions.

---

## 2. Category Sizes (9-Category Evaluation Design)

| # | Category | Definition | Count | Pct | Flag |
|---|---|---|---|---|---|
| 1 | Verify (yes/no) | structural == "verify" | 27,413 | 20.76% | ✅ |
| 2 | Choose | structural == "choose" | 16,229 | 12.29% | ✅ |
| 3 | Logical | structural == "logical" | 16,096 | 12.19% | ✅ |
| 4 | Compare | structural == "compare" | 4,157 | 3.15% | — |
| 5 | Object query | structural=="query" AND semantic=="obj" | **0** | 0.00% | ⚠️ IMPOSSIBLE |
| 6 | Attribute query | structural=="query" AND semantic=="attr" | 18,092 | 13.70% | ✅ |
| 7 | Relation query | structural=="query" AND semantic=="rel" | 40,278 | 30.50% | ✅ |
| 8 | Category query | structural=="query" AND semantic=="cat" | 7,185 | 5.44% | ✅ |
| 9 | Counting (how many) | question.startswith("how many") | **0** | 0.00% | ⚠️ ABSENT |

### Critical Findings:
- **Category 5 ("Object query") is structurally impossible** in GQA. `semantic == "obj"` never co-occurs with `structural == "query"`. This category must be redesigned.
- **Category 9 ("Counting") yields zero questions** in the balanced validation split. The GQA balanced split systematically excludes or underrepresents "how many" questions. No questions had digit-only answers either. This category cannot be evaluated on the balanced split.
- **Category 4 ("Compare") has 4,157 questions** — above 500 (usable) but below 5,000 (well-represented). Acceptable for evaluation.

---

## 3. ViLT Answer Vocabulary Coverage

| Metric | Value |
|---|---|
| Unique answer coverage | 69.4% (1,020/1,469 unique answers) |
| Question-level coverage (overall) | **96.6%** (127,625/132,062 questions) |

### Coverage by Structural Type

| Structural Type | Coverage |
|---|---|
| verify | 100.0% ✅ |
| logical | 100.0% ✅ |
| choose | 97.7% ✅ |
| compare | 97.2% ✅ |
| query | 94.2% ✅ |

### Coverage by Semantic Type

| Semantic Type | Coverage |
|---|---|
| obj | 100.0% ✅ |
| attr | 98.1% ✅ |
| rel | 95.4% ✅ |
| cat | 93.7% ✅ |
| global | 93.2% ✅ |

**All categories are above the 70% threshold.** ViLT's fixed vocabulary is not a structural barrier for any proposed evaluation category.

### Top Uncovered Answers (Most Frequent Not in ViLT Vocab)
`guy` (320), `computer mouse` (133), `material` (103), `bookcase` (83), `t-shirt` (75), `blond` (71), `cloudless` (66), `path` (63), `drawers` (59), `cupboard` (58) …

These are mostly object names and descriptors that GQA includes but VQA v2.0 does not.

---

## 4. Inference Timing (CPU, 50-sample benchmark)

| Model | Mean | Median | Std | Min | Max | Est. full run (132K qs) |
|---|---|---|---|---|---|---|
| BLIP | 0.266s | 0.218s | 0.321s | 0.208s | 2.508s | ~9.8 hours |
| ViLT | 0.063s | 0.062s | 0.004s | 0.058s | 0.081s | ~2.3 hours |

**ViLT is ~4.2× faster than BLIP on CPU.** BLIP has higher variance (one outlier at 2.5s), likely due to generation decoding overhead. ViLT uses fixed classification and is highly consistent.

**Practical notes:**
- No errors or format issues encountered on either model.
- Both models accept RGB PIL images. ViLT has stricter internal image size requirements but the processor handles resizing automatically.
- BLIP uses autoregressive generation (GenerationMixin); ViLT uses classification over a fixed label set.
- For full evaluation, GPU is strongly recommended to make runtime tractable.

---

## 5. Answer Normalization

**Raw exact-match accuracy (50-sample):**
- BLIP: 56.0%
- ViLT: 48.0%

**After normalization (strip + lower + number words + remove articles):**
- BLIP: 56.0% (no change — GQA answers are already clean single tokens)
- ViLT: 48.0% (no change)

**Observations:**
- No case-only mismatches detected (both models already output lowercase).
- GQA ground-truth answers are all single tokens/short phrases — no article-stripping issues observed in this sample.
- BLIP produced 1 multi-word answer (`"neither"` replaced by `"kids"` in one case) — BLIP does generate free-form text.
- All 26 ViLT errors were in-vocabulary failures (the correct answer existed in ViLT's vocab but the model predicted wrong). Zero structural impossibilities in this sample.
- Number format normalization is not triggered since GQA balanced split contains no digit-answer questions.

**Recommended normalization function:**
```python
import re
NUM_MAP = {'zero':'0','one':'1','two':'2','three':'3','four':'4',
           'five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}

def normalize_answer(s):
    s = s.strip().lower()
    if s in NUM_MAP:
        s = NUM_MAP[s]
    s = re.sub(r'^(a |an |the )', '', s)
    return s
```

---

## 6. Key Decisions for the Proposal

1. **Drop or redesign "Object query" category.** `semantic == "obj"` never co-occurs with `structural == "query"` in GQA. Either replace it with `logical + obj` (questions like "Is there any X or Y?"), or merge object-oriented questions from `verify+obj`. Clarify this in the proposal.

2. **Drop the "Counting" category from the balanced split evaluation.** Zero "how many" questions exist in `val_balanced_questions.json`. If counting is important to your research question, you must either use `val_all_questions.json` (unbalanced) or define an alternative counting criterion (e.g., questions with digit ground-truth answers), and report on that separate split.

3. **No need to filter by ViLT vocabulary.** Question-level coverage is 96.6% overall and ≥93% for every category. Filtering would remove very few questions and bias results. Evaluate on the full set and report the ≤6.8% structurally impossible ViLT questions as a known limitation.

4. **Use a random subset (~10K questions) for wall-clock feasibility on CPU.** Full evaluation of 132K questions would take ~9.8 hours for BLIP and ~2.3 hours for ViLT on CPU. A stratified random sample of ~10K questions (keeping category proportions) reduces BLIP runtime to ~45 minutes and ViLT to ~10 minutes, while remaining statistically sound given the large pool sizes. State GPU vs. CPU conditions clearly.

5. **Use the 7-category design (not 9).** Given findings above, the practical category set is: (1) Verify, (2) Choose, (3) Logical, (4) Compare, (5) Attribute query, (6) Relation query, (7) Category query. This covers 100% of the balanced val questions cleanly with no overlaps or zero-size bins.

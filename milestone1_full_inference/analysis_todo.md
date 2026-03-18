# NLP Project: Analysis Concerns & Next Steps
**Project:** Question-Type-Aware VQA Evaluation: BLIP vs. ViLT on GQA
**Status:** Milestone 1 (full inference) complete. These are the remaining checks and analyses needed before drawing final conclusions.

---

## Context: What We Have

- `predictions/all_predictions.jsonl` — raw predictions for all 132,062 questions, both models
- `results/per_question_stats.csv` — 132,062 rows with strict/relaxed correctness for both models
- `results/blip_mismatch_analysis.csv` — 5,152 BLIP "correct-ish" cases
- `results/strict/` and `results/relaxed/` — full results under both evaluation modes
- GQA annotation file `val_balanced_questions.json` — contains `structural`, `semantic`, and `semantic` (functional program) fields per question

---

## Block 1: Evaluation Sanity Checks (Must Do First)

These need to be resolved before any conclusions about model performance are valid.

### 1.1 ViLT OOV Breakdown

**Concern:** ViLT can only output one of its fixed 3,129-label vocabulary. If the ground-truth answer is not in that vocab, ViLT literally cannot answer correctly — this is a vocabulary artifact, not a reasoning failure. Currently we cannot tell whether ViLT's errors come from (A) vocab gaps or (B) genuine perceptual/reasoning failures.

**What to do:**
- Load ViLT's fixed vocabulary (extractable from `dandelin/vilt-b32-finetuned-vqa` config)
- For every ViLT wrong prediction in `per_question_stats.csv`, check: is the GT answer in the vocab or not?
- Compute per-cell breakdown:
  - `n_oov` = questions where GT is not in vocab (Type A error — unavoidable)
  - `n_in_vocab_wrong` = questions where GT is in vocab but ViLT predicted wrong (Type B — genuine error)
  - `true_error_rate` = `n_in_vocab_wrong / (total_questions - n_oov)`
- Report this breakdown per cell in the 5×5 matrix
- **Expected impact:** `query×cat` and `query×rel` cells are most likely to be affected, since GQA includes object/relation labels that VQA v2 does not

**Why this matters:** The BLIP−ViLT gap in some cells might shrink or disappear once OOV questions are excluded. This is essential before attributing any gap to architecture.

---

### 1.2 BLIP Exact-Match Error Categorization

**Concern:** BLIP generates free text. Exact-match penalizes it for surface mismatches that may be semantically correct. We already have 5,152 "correct-ish" cases in `blip_mismatch_analysis.csv`, but need to decide which categories are legitimate corrections.

**What to do:**
- Review `blip_mismatch_analysis.csv` category by category:
  - `blip_superset` (3,004): GT="shirt", BLIP="pink shirt" — BLIP is more specific; arguably correct
  - `blip_subset` (1,959): GT="white shirt", BLIP="shirt" — BLIP missed the attribute; arguably wrong
  - `plural` (1,335): GT="cat", BLIP="cats" — unambiguously should be correct
  - `prep_prefix` (281): GT="snow", BLIP="on snow" — arguably equivalent
  - `synonym_overlap` (189): GT="sofa", BLIP="couch" — genuinely correct, penalized unfairly
- Decide on a "relaxed-v2" evaluation: which categories to include beyond plural+prep_prefix
- Compute per-cell accuracy under each version (strict / relaxed-v1 / relaxed-v2)
- Check where in the 5×5 matrix these corrections concentrate (should be ~95% in `query` row)

**Why this matters:** `query` row accuracy for BLIP is currently understated. The true gap over ViLT on S1 (open-ended retrieval) is likely higher than reported. Conversely, if `blip_subset` cases are excluded from corrections, the adjustment is smaller.

---

### 1.3 Verify the `choose` Anomaly Mechanism

**Concern:** The largest BLIP advantage is on `choose` questions (`choose×attr` +.103, `choose×cat` +.122). Our hypothesis is that ViLT's fixed classifier is blind to the two options presented in the question, while BLIP's decoder naturally conditions on them. But this needs to be verified directly.

**What to do:**
- From `all_predictions.jsonl`, extract all `choose` questions where ViLT is wrong
- For each, check: is ViLT's predicted answer one of the two options offered in the question, or is it something else entirely?
- Compute: `% of ViLT wrong choose predictions where predicted label is NOT one of the two options`
- Do the same for BLIP wrong on `choose`: does BLIP tend to at least pick one of the two options when wrong?
- A random sample of 50–100 cases is sufficient; this can be manual inspection or scripted

**Expected finding:** ViLT frequently outputs labels outside the offered pair. BLIP rarely does. If confirmed, this is the strongest architectural finding in the paper.

**Why this matters:** This is the one finding most likely to survive the pretraining-scale confound. If ViLT's errors on `choose` are systematically "wrong option" answers (not one of the two offered), that's a clear architectural failure of fixed classification.

---

## Block 2: Depth Analysis (Priority 2)

### 2.1 Program Depth Distribution and Accuracy

**Concern:** The GQA paper itself showed accuracy drops with program depth (number of reasoning operations). We want to know: (a) does this hold for BLIP and ViLT, and (b) does the BLIP−ViLT gap grow at higher depths (which would suggest BLIP's decoder handles complex reasoning better)?

**What to do:**
- Join `per_question_stats.csv` with `val_balanced_questions.json` on question ID
- Compute `program_depth = len(q['semantic'])` for each question
- Assign depth bins: {2, 3, 4, 5, ≥6}
- Plot accuracy vs. depth bin for both models (line chart, one line per model)
- Compute and plot the BLIP−ViLT gap vs. depth bin

**Critical: control for structural-type confound**
- Depth and structural type are correlated: `logical` has mean depth 5.32, all others ≤3.0
- High-depth questions are mostly `logical×obj`, which are always yes/no (50% random baseline)
- To isolate the depth effect from the type effect, also plot accuracy vs. depth *within each structural type* separately:
  - Within `query`: depth 2 vs. 3 vs. 4
  - Within `logical`: depth 4 vs. 5 vs. 6 vs. 7
  - Within `verify`: depth 2 vs. 3

**What to report:**
- Overall accuracy vs. depth plot (with caveat about confound)
- Per-structural-type accuracy vs. depth
- Whether the BLIP gap widens, stays constant, or narrows with depth

---

### 2.2 Operation-Type Analysis (Optional, Lower Priority)

**Concern:** Raw depth is a coarse measure. The type of operations in the program (e.g., `relate` vs. `filter` vs. `and`) may be more predictive of errors than depth alone.

**What to do (if time permits):**
- Extract operation types from each question's `semantic` field
- Flag questions by presence of: `relate` operations, `filter` operations, `and`/`or` operations
- Compute accuracy for each flag group per model
- Particularly interesting: questions with `relate` ops (spatial grounding) vs. without

---

## Block 3: Error Pattern Analysis (Priority 3)

### 3.1 Top Wrong Predictions Per Cell

**Concern:** We don't know yet whether errors are random or systematic. Systematic errors (e.g., ViLT always predicting "yes" on logical questions, or always predicting the same color on color questions) would reveal biases that go beyond simple accuracy numbers.

**What to do:**
- For each of the 15 cells, for each model, compute the top-10 most frequent wrong predictions
- Look for: (a) mode-seeking behavior (classifier collapses to a few labels), (b) yes/no bias on verify/logical, (c) BLIP generating the same generic phrases repeatedly
- Specifically check `logical` rows: is ViLT or BLIP biased toward "yes" or "no"?
- Specifically check `compare×attr`: since both models tie at ~.487, do they make the same errors or different ones?

**Output:** A table per cell showing top wrong predictions and their frequencies.

---

### 3.2 Duplicate Question Check (Minor Sanity Check)

**Concern:** GQA generates multiple phrasings of the same underlying graph query (e.g., "What is in front of the giraffe?" and "What's in front of the giraffe?" are semantically identical). A model that answers one correctly and one wrongly has surface-form sensitivity.

**What to do:**
- Identify duplicate question pairs in the validation set (same image ID + same answer + very similar question text)
- Check: do BLIP and ViLT give consistent predictions on these pairs?
- High inconsistency would be a separate finding about robustness

**Note:** This is lower priority — mention in limitations if not done.

---

## Block 4: Known Confounds to Document (No Code Needed)

These are limitations to address in the write-up rather than analyze further.

### 4.1 Pretraining Scale Confound
BLIP and ViLT differ in four ways simultaneously:
1. Fusion strategy (late cross-attention vs. early unified)
2. Answer mechanism (generative vs. fixed classifier)
3. Pretraining scale (~130M pairs vs. ~4.4M pairs, ~30× difference)
4. Fine-tuning data (VQA v2 + Visual Genome vs. VQA v2 only)

The uniform +2–4pp BLIP advantage across most cells is most consistent with pretraining scale being dominant. The `choose` anomaly (+10–12pp) is the strongest candidate for a genuine architectural/answer-mechanism effect. These cannot be fully disentangled without controlled ablations.

### 4.2 Zero-Shot Transfer on GQA
Both models were fine-tuned on VQA v2, not GQA. Running them on GQA is effectively zero-shot transfer. Some errors may come from domain shift (GQA answer vocabulary differs from VQA v2) rather than model weaknesses. The OOV analysis (Block 1.1) partially addresses this for ViLT.

### 4.3 GQA Exact-Match Strictness
The GQA paper itself acknowledged that single-GT exact match underestimates model performance because semantically correct paraphrases are penalized. This is a known dataset-level limitation, not specific to our evaluation. Cite the paper's acknowledgment of this issue.

### 4.4 Data Quality Issues in GQA
Some GQA questions have known quality issues (from `image_case_study.md`):
- Age comparison questions (e.g., "Who is older, the lady or the boy?") cannot be reliably answered from visual features alone — the answer is derived from object labels (lady vs. boy), not visual appearance
- Duplicate questions (same semantic content, different phrasing) inflate some cells

---

## Summary: Recommended Execution Order

| Step | Task | Data Needed | Output |
|---|---|---|---|
| 1 | ViLT OOV breakdown | `all_predictions.jsonl` + ViLT vocab | Per-cell OOV rate + true error rate |
| 2 | BLIP mismatch review | `blip_mismatch_analysis.csv` | Decide relaxed-v2 evaluation scope |
| 3 | Verify `choose` mechanism | `all_predictions.jsonl` | % of ViLT wrong choose answers outside offered pair |
| 4 | Program depth analysis | `per_question_stats.csv` + `val_balanced_questions.json` | Accuracy vs. depth plots, per structural type |
| 5 | Top wrong predictions | `all_predictions.jsonl` | Per-cell top wrong prediction tables |
| 6 | Write-up | All above | Results + discussion section |

Steps 1–3 are sanity checks that may change the interpretation of existing results. Steps 4–5 are new analyses on existing inference data. No new model inference is required for any of these.

---

## Key Questions That Should Be Answerable After the Above

1. How much of ViLT's lower accuracy in `query×cat` and `query×rel` is vocabulary gap vs. actual reasoning failure?
2. Is ViLT's failure on `choose` questions specifically due to predicting labels outside the offered pair? (This would confirm the architectural mechanism.)
3. Does the BLIP−ViLT gap grow with program depth, or is it flat? (Flat = pretraining scale; widening = architectural signal.)
4. Are errors on `compare×attr` (the only tie cell) the same errors or different errors for each model?
5. After OOV correction, what is ViLT's "true" per-cell accuracy on questions it could theoretically answer?
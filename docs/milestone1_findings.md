# Milestone 1 — Report Writing Reference
> Last updated: 2026-03-19
> Status: **FINAL** — all numbers are post-fix (normalization rules 1-6 applied,
> OOV analysis complete, BLIP mismatch investigated and resolved).
> Organized to mirror the milestone report sections.
> Quick-reference to figures: see §F.

---

## A. Dataset & Taxonomy
*Report section: "Dataset" or "Experimental Setup". Details → appendix.*

### A1. The 5×5 taxonomy grid

GQA questions are classified on two independent axes:
- **Structural** (reasoning form): query, verify, logical, choose, compare
- **Semantic** (visual target): rel, attr, obj, cat, global

15 of 25 cells are populated in the balanced val split (132,062 questions):

| | rel | attr | obj | cat | global |
|---|---|---|---|---|---|
| **query** | 40,278 | 18,092 | — | 7,185 | 2,612 |
| **verify** | 15,602 | 8,053 | 2,879 | — | 879 |
| **logical** | — | 3,411 | 12,685 | — | — |
| **choose** | 5,754 | 8,488 | — | 1,431 | 556 |
| **compare** | — | 4,157 | — | — | — |

### A2. Answer vocabulary (important for evaluation design)

Each question has exactly **1 GT answer string** (no multi-annotation in GQA).
The answer vocabulary varies dramatically across cells:

| Cell type | # unique answers | Character |
|-----------|-----------------|-----------|
| verify×\*, logical×\* | 2 | always yes/no |
| compare×attr | 46 | 78.4% yes/no — behaves like binary |
| choose×rel | 110 | "left"/"right" dominate (90%) |
| choose×attr | 222 | positional + color terms |
| query×attr | 127 | colors, materials, positions |
| query×cat | 370 | object categories |
| query×global | 93 | scene/place names |
| query×rel | **1,199** | people, objects, relations — highest open-vocab |

**Key implication**: answer type (binary vs constrained-choice vs open-vocab) is a
stronger predictor of model difficulty than the structural or semantic axis label.
This is a core observation for the report. See §D1.

### A3. Logic depth per cell

"Logic depth" = number of reasoning steps in the GQA program graph.

**Notable patterns:**
- **Always depth-2 (zero complexity variance)**: query×global, choose×global, verify×global
- **Logical×obj bimodal**: splits at depth 5 (n=10,585) and depth 7 (n=3,506) — these
  are structurally different question subtypes within the same cell label
- **logical×attr**: depth 4–6 only (mean 4.45), confirming compound reasoning
- **verify×rel**: widest spread (depth 2–5), suggesting this cell is heterogeneous
- **logical** has the highest mean depth (5.32) vs ≤3.0 for all other structural types

**For report**: define "logic depth" early (it is GQA's program graph depth, not
sentence length). Put depth histograms per cell in appendix; cite key numbers in text.

### A4. Scene graph availability
- 10,234 unique images; avg 16.3 objects, 50.0 relations, 9.0 attributes per image
- 82.2% of questions have scene-graph annotations linking question tokens → object IDs
- Available for future error analysis (accuracy vs. object richness, occlusion, etc.)

---

## B. Challenges & Evaluation Decisions
*Report section: "Challenges" or "Evaluation Design". Reference `evaluation_decisions.md`
for full reasoning; this section has the summary for report writing.*

### B1. ViLT OOV — **Investigated and resolved**

**The problem**: ViLT uses a fixed 3,129-label classifier (the VQA v2 answer vocabulary).
Any GQA GT answer outside this set is structurally impossible to predict — not a
reasoning failure. This could make ViLT look artificially worse than it is.

**Why 3,129 labels**: these are the most frequent answers in VQA v2 training data,
covering single-word answers for common colors, objects, yes/no, positions, numbers.
Multi-word answers (e.g., "grey rabbit") are absent even if individual words appear.

**What we found**:

| Metric | Value |
|--------|-------|
| OOV questions (raw GT) | 3,998 / 132,062 (3.0%) |
| OOV questions after normalization | ~3,000 / 132,062 (~2.3%) |
| Most affected cells | query×rel (6.0%), query×global (10.2%), query×cat (6.3%) |
| BLIP accuracy on OOV questions | 6.0% (rare GT answers are hard for BLIP too) |
| ViLT accuracy on OOV questions | 0.0% (structural ceiling) |

**Comparison — does OOV explain the BLIP vs ViLT gap?**

| Subset | BLIP | ViLT | Gap |
|--------|------|------|-----|
| Full dataset — strict | 52.9% | 50.0% | +2.9pp |
| Full dataset — normalized | 53.9% | 50.9% | +3.0pp |
| Answerable only (OOV excluded) — normalized | 55.4% | 52.5% | +2.9pp |

**Conclusion**: OOV exclusion changes the gap by only **0.1pp**. The ViLT deficit is
genuine, not a vocabulary artifact. We still report normalized-OOV-excluded as a
methodological transparency measure, not as the primary metric.

**For report**: frame as "we verified this is not an artifact — OOV accounts for 0.1pp
of the 3pp gap." Show the 3-row comparison table above.

---

### B2. BLIP answer mismatch — **Investigated and resolved via normalization**

**The problem**: BLIP generates free-form text. Strict exact-match penalizes legitimate
surface-form variation (e.g., "bricks" vs "brick", "gray" vs "grey", "2" vs "two").

**What we investigated**: ran a mismatch detector on all BLIP incorrect predictions to
find "correct-ish" failures. Found 5,152 such cases across structural types.

**Findings by category**:

| Structural | Correct-ish cases | Verdict |
|------------|------------------|---------|
| choose | 223 | **All wrong** — choose questions explicitly provide two candidate answers; BLIP should select one of them, not generate its own form |
| verify | 0 | Clean — yes/no has no surface variation |
| logical | 16 | **All false positives** — detector bug: "no" ⊂ "snow", "no" ⊂ "notes" (substring accidents) |
| compare | 20 | **Mostly false positives** — "man" ⊂ "woman" substring accident; a few borderline superset cases |
| query | 4,893 | **Two genuine patterns worth fixing** (see below) |

**Genuine mismatch patterns in query (the only category that matters)**:

| Pattern | Example | Count | Resolution |
|---------|---------|-------|------------|
| Plural/singular | GT=`brick` BLIP=`bricks` | ~1,330 | ✅ Fixed by rule 6 (depluralize) |
| British/American spelling | GT=`blond` BLIP=`blonde` | ~150 | ✅ Fixed by adding `blonde→blond` to rule 5 |
| BLIP color over-specification | GT=`blue` BLIP=`blue and white` | ~600 | ❌ Genuinely wrong — BLIP fails to identify the single annotated color |
| BLIP drops color modifier | GT=`dark blue` BLIP=`blue` | ~700 | ❌ Genuinely wrong — BLIP cannot distinguish shade precision |

**Conclusion**: After applying normalization rules 1–6 (including the `blonde→blond`
fix), the residual mismatch problem is limited to genuine model errors, not surface-form
artifacts. No additional normalization is needed.

**For report**: frame as "we investigated generative mismatch — the residual errors after
normalization are genuine model failures, not annotation artifacts." This demonstrates
methodological rigor without overstating the problem.

---

### B3. Pretraining asymmetry — **Acknowledged limitation**

BLIP (generative encoder-decoder) and ViLT (fixed-label classifier) differ in
pretraining objectives, answer mechanisms, and model size. They cannot be perfectly equalized.

**How to frame in the report**:
1. Acknowledge explicitly — this is not a controlled ablation
2. Our contribution is NOT "BLIP beats ViLT" — it is that **per-type accuracy varies
   substantially in ways that aggregate accuracy hides**, and that this variation is
   systematic and interpretable
3. The normalized-OOV-excluded evaluation gives the fairest comparison within the
   architectural constraints we have
4. Both models are always evaluated on identical question sets

---

## C. Results
*Report section: "Results". Primary figures: heatmaps + depth line graphs.*

### C1. Overall accuracy

| Mode | BLIP | ViLT | Gap |
|------|------|------|-----|
| Strict (raw exact-match) | 52.9% | 50.0% | +2.9pp |
| Normalized (rules 1–6) | 53.9% | 50.9% | +3.0pp |
| Normalized-OOV-excluded | 55.4% | 52.5% | +2.9pp |

Normalization lifts BLIP by 1.0pp and ViLT by 0.9pp — comparable benefit for both
models, confirming no systematic bias in the normalization rules.

### C2. Per-cell accuracy (normalized)

Sorted by cell size. **Bold** = noteworthy; see §D for discussion.

| structural | semantic | n | BLIP | ViLT | gap |
|------------|----------|---|------|------|-----|
| query | rel | 40,278 | 0.434 | 0.419 | +0.016 |
| query | attr | 18,092 | 0.516 | 0.494 | +0.021 |
| verify | rel | 15,602 | 0.722 | 0.684 | +0.038 |
| logical | obj | 12,685 | 0.514 | 0.489 | +0.025 |
| choose | attr | 8,488 | 0.654 | 0.557 | **+0.097** |
| verify | attr | 8,053 | 0.672 | 0.650 | +0.022 |
| query | cat | 7,185 | 0.567 | 0.532 | +0.035 |
| choose | rel | 5,754 | 0.521 | 0.508 | +0.013 |
| compare | attr | 4,157 | 0.499 | 0.501 | **−0.002** |
| logical | attr | 3,411 | 0.708 | 0.672 | +0.036 |
| verify | obj | 2,879 | 0.786 | 0.740 | +0.046 |
| query | global | 2,612 | **0.387** | **0.344** | +0.043 |
| choose | cat | 1,431 | 0.751 | 0.638 | **+0.113** |
| verify | global | 879 | **0.956** | **0.951** | +0.005 |
| choose | global | 556 | **0.956** | **0.910** | +0.046 |

**Ceiling cells** (both models >90%): verify×global, choose×global — binary yes/no with
a near-deterministic scene-level answer ("indoors"/"outdoors").

**Floor cells** (both models <45%): query×rel (open-vocab, 1,199 unique answers),
query×global (open-vocab place names).

**Largest BLIP lead**: choose×cat (+11.3pp). Likely ViLT OOV on specific category names.

**Only ViLT lead**: compare×attr (−0.002pp, essentially tied). Near-binary answers
mean both models are near-chance; ViLT's slight "no" bias accidentally helps here.

### C3. Accuracy vs Logic Depth (normalized)

**Aggregate is misleading** — do not show a single pooled line in the report.
Instead use the per-structural-type breakdown.

**Overall pooled (for context only)**:
| depth | n | BLIP | ViLT | note |
|-------|---|------|------|------|
| 2 | 36,743 | 0.622 | 0.584 | |
| 3 | 62,810 | 0.501 | 0.475 | |
| 4 | 17,875 | 0.512 | 0.483 | |
| 5 | 11,036 | 0.599 | 0.551 | ⚠ inflated — 83% logical×obj (binary yes/no) |
| 6 | 85 | 0.459 | 0.494 | ViLT > BLIP; logical×obj compound (sparse) |
| 7 | 3,506 | 0.307 | 0.348 | ViLT > BLIP; logical×obj compound AND/OR |

**Per structural type — use this in the report**:

| structural | depth | n | BLIP | ViLT | gap |
|------------|-------|---|------|------|-----|
| query | 2 | 14,821 | 0.502 | 0.478 | +0.024 |
| query | 3 | 42,149 | 0.444 | 0.424 | +0.020 |
| query | 4 | 10,891 | 0.387 | 0.366 | +0.021 |
| verify | 2 | 12,905 | 0.732 | 0.708 | +0.024 |
| verify | 3 | 9,559 | 0.721 | 0.678 | +0.043 |
| verify | 4 | 4,805 | 0.698 | 0.659 | +0.039 |
| logical | 4 | 1,913 | 0.742 | 0.700 | +0.042 |
| logical | 5 | 10,585 | 0.604 | 0.557 | +0.047 |
| logical | 7 | 3,506 | **0.307** | **0.348** | **−0.041** |
| choose | 2 | 8,548 | 0.666 | 0.581 | +0.085 |
| choose | 3 | 7,414 | 0.560 | 0.506 | +0.055 |
| compare | 2 | 469 | 0.642 | 0.582 | +0.060 |
| compare | 3 | 3,688 | 0.466 | 0.476 | −0.009 |

---

## D. Observations & Insights
*Report section: "Analysis" or "Discussion". These are the interpretive findings.*

### D1 — Answer type is the dominant predictor (stronger than structural/semantic type)

The structural axis conflates two independent variables:
- **Answer type**: binary (yes/no) vs constrained-choice vs open-vocabulary
- **Reasoning form**: attribute lookup vs relation traversal vs logical combination

**Evidence**: verify×global (95.6%) vs query×global (38.7%). Same semantic target, same
image, different answer type. The semantic label ("global") predicts nothing; answer type
explains everything.

**Implication for the taxonomy**: the 5×5 structural×semantic grid is a reasonable
*description* of the dataset but a weak *predictor* of model difficulty. Answer type
(derivable from the cell) is the actual signal.

**For report**: lead with this observation. It motivates why type-aware analysis reveals
structure that aggregate accuracy hides, and sets up the future work on explicit
answer-type decomposition.

---

### D2 — compare×attr is taxonomically near-binary

78.4% of compare×attr answers are yes/no. Its accuracy (BLIP 49.9%, ViLT 50.1%)
matches binary-answer cells, not what "comparison" implies. Both models hover near
chance with a slight "no" bias. Flag in the report as a structural ambiguity in the
GQA taxonomy — this cell is harder to interpret than its label suggests.

---

### D3 — Aggregate depth accuracy is misleading (depth-5 artifact)

The pooled depth-5 accuracy (BLIP 59.9%) is *higher* than depth-3 (50.1%) and
depth-4 (51.2%). This is **not** because models handle complex reasoning better —
it is because depth-5 is 83% logical×obj (binary yes/no), which inflates the average.

**The per-type depth story**:
- **query declines steadily**: 50.2% → 44.4% → 38.7% (depths 2→3→4). Longer chains
  genuinely hurt open-vocab retrieval.
- **verify is nearly flat**: 73.2% → 72.1% → 69.8%. Binary format absorbs complexity.
- **logical shows the largest cliff**: 60.4% (depth-5) → 30.7% (depth-7). A 30pp drop
  is the most dramatic depth effect in the dataset.

**For report**: show the 5-panel per-structural-type figure
(`depth_accuracy_by_structural.png`), not the aggregate. Use the aggregate only as a
motivating example of why pooling is misleading.

---

### D4 — ViLT > BLIP at logical depth-7 (the only model reversal)

At depth-7 logical×obj (compound AND/OR questions, n=3,506):
- BLIP: 30.7% — below ViLT
- ViLT: 34.8%
- Gap: **−4.1pp** (ViLT leads)

This is the only cell×depth combination where ViLT consistently beats BLIP.
Hypothesis: BLIP's generative decoder over-generates on deeply nested binary questions,
producing multi-word answers ("there is a dog") that fail exact-match even after
normalization. ViLT's classifier directly outputs "yes"/"no" and is structurally
immune to this failure mode.

---

### D5 — Logical×obj bimodal: two structurally distinct subtypes in one cell

`logical×obj` splits cleanly by depth, suggesting the cell conflates two question types:

| Depth | n | Structure | BLIP | ViLT | Gap |
|-------|---|-----------|------|------|-----|
| 5 | 10,585 | Single relate→verify chain ("Is there a man near the car?") | 0.604 | 0.557 | +0.047 |
| 7 | 3,506 | Compound AND/OR of two sub-questions ("Is there a dog or a cat?") | 0.307 | 0.348 | **−0.041** |

The 30pp accuracy cliff and the model reversal at depth-7 are the most striking
depth-related findings. This is a good example for the "limitations of the current
taxonomy" discussion and future work section.

---

### D6 — choose×cat has the largest BLIP lead (+11.3pp)

BLIP leads ViLT by 11.3pp on choose×cat (n=1,431). These are constrained-choice
object-category questions ("Is it a dog or a cat?"). ViLT's fixed-vocab classifier
struggles when the correct answer is a specific category name that appears infrequently
in VQA v2 training data. Partially explained by OOV (3.6% of choose×cat questions are
OOV), but the gap persists even in the answerable subset (+11.3pp → see C2 answerable).

---

## E. Future Work
*Report section: "Future Work" or "Remaining Experiments".*

1. **Answer-type decomposition**: Re-analyze accuracy with answer-type as primary axis
   (binary / constrained-choice / open-vocab) rather than structural type. This tests
   whether answer type is a better predictor than the structural label (Insight D1).

2. **Within-cell depth analysis**: For cells with depth variance (verify×rel depth 2–5,
   logical×obj depth 5 vs 7), plot accuracy vs depth bin. Already partially done;
   extend to all cells and frame as "within-cell complexity analysis."

3. **OOV deep-dive per cell**: Quantify per cell what fraction of ViLT errors are
   OOV-impossible vs genuine reasoning failures. This separates architectural ceiling
   from actual model competence.

4. **Scene graph correlation**: Does accuracy drop on questions referencing objects
   with fewer scene-graph attributes or more scene clutter? Requires joining
   `per_question_stats.csv` with `val_sceneGraphs.json`.

5. **Taxonomy revision**: Consider replacing the structural axis with an answer-type
   axis for cleaner experimental signal. Evaluate whether the 5×5 grid is the right
   decomposition for studying model reasoning.

6. **Error mode analysis for logical depth-7**: Examine BLIP's actual outputs on
   compound AND/OR questions. Are they consistently multi-word phrases? Does
   semantic relaxation (Type 2 matching) recover the gap?

---

## F. Figures Inventory
*All figures are generated and current as of 2026-03-19.*

### Heatmaps (`results/strict/`, `results/normalized/`, `results/normalized_OOV_excluded/`)
Each directory contains:
- `accuracy_heatmap_blip.png` — BLIP accuracy 5×5 heatmap
- `accuracy_heatmap_vilt.png` — ViLT accuracy 5×5 heatmap
- `accuracy_heatmap_gap.png` — BLIP−ViLT gap heatmap

**For report**: use the `normalized/` versions as primary. Show `strict/` as baseline in
appendix or supplementary. Show `normalized_OOV_excluded/` gap heatmap in the challenges
section to demonstrate OOV impact is minimal.

### Depth accuracy (`results/analysis/depth_accuracy/`)
| File | Use in report |
|------|--------------|
| `depth_accuracy_overall.png` | Motivating example: show the pooled line, note the depth-5 artifact |
| `depth_accuracy_by_structural.png` | **Main figure** — 5-panel per structural type |
| `depth_accuracy_logical_detail.png` | Supporting figure — bimodal split in logical×obj |
| `depth_accuracy_heatmap.png` | Supplementary / appendix |

### EDA / exploration (`results/exploration/`)
| Directory | Contents | Report use |
|-----------|----------|-----------|
| `answer_structure/` | Vocab size heatmap, overlap heatmap, answer length plots, annotated examples | Appendix — dataset characterization |
| `depth_distribution/` | Depth histograms, mean depth heatmap, example questions per depth | Appendix — depth analysis |
| `dataset_fields/` | Scene graph stats, field guide, detailed type distribution | Appendix or supplementary |

### OOV analysis (`results/exploration/vilt_oov/`)
- `oov_analysis_report.txt` — summary stats (OOV before/after normalization, per-cell)
- `figures/oov_rate_heatmap.png` — OOV rate per cell
- For report: cite top-line numbers from §B1; put heatmap in challenges section.

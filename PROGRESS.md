# Project Progress & Findings
**Question-Type-Aware VQA Analysis: BLIP vs. ViLT on GQA**
_Last updated: milestone 1 complete_

---

## 1. What We Are Doing

An NLP course project evaluating two pretrained VQA models — **BLIP** and **ViLT** — on the **GQA balanced validation split** (132,062 questions) using GQA's native two-axis question taxonomy. The core claim is that aggregate VQA accuracy hides important per-type variation. We expose that variation through a **5×5 evaluation matrix** crossing structural type × semantic type, and a secondary **program depth analysis** (planned, not yet run).

---

## 2. Models

| | **ViLT** | **BLIP** |
|---|---|---|
| HuggingFace ID | `dandelin/vilt-b32-finetuned-vqa` | `Salesforce/blip-vqa-base` |
| Architecture | Single unified transformer (encoder-only) | Encoder + causal decoder (MED) |
| Image encoder | ViT-B/**32** (patch=32, res=384) | ViT-B/**16** (patch=16, res=480 for VQA) |
| VQA output | Fixed 3,129-class softmax classifier | Open-vocab generative decoding (max 20 tokens) |
| Pretraining data | CC3M + SBU + COCO + VG ≈ **~4.4M pairs** | CC3M + CC12M + SBU + COCO + VG + LAION115M ≈ **~130M pairs** |
| Pretraining objectives | MLM + ITM + WPA | ITC + ITM + LM + CapFilt bootstrapping |
| Fine-tuning data | VQA v2 only | VQA v2 + Visual Genome QA |
| Parameters | ~87M | ~224–250M |

**Key confound:** BLIP and ViLT differ in both their answer mechanism (generative vs. classifier) *and* their pretraining scale (~30× more data), patch resolution (4× finer), and fine-tuning data. These factors cannot be disentangled from the current comparison alone.

---

## 3. Dataset

- **GQA balanced validation split**: 132,062 questions, 10,234 unique images
- **Two annotation axes per question:**
  - *Structural type* (reasoning form): `query`, `verify`, `logical`, `choose`, `compare`
  - *Semantic type* (visual target): `rel`, `attr`, `obj`, `cat`, `global`
- **15 populated cells** in the 5×5 matrix (10 cells are structural zeros)
- Every question also has a **functional program** (DAG of reasoning steps) — program depth is a built-in complexity measure requiring no extra annotation

### 5×5 Cell Counts

|  | rel | attr | obj | cat | global | **Total** |
|---|---:|---:|---:|---:|---:|---:|
| **query** | 40,278 | 18,092 | — | 7,185 | 2,612 | 68,167 |
| **verify** | 15,602 | 8,053 | 2,879 | — | 879 | 27,413 |
| **logical** | — | 3,411 | 12,685 | — | — | 16,096 |
| **choose** | 5,754 | 8,488 | — | 1,431 | 556 | 16,229 |
| **compare** | — | 4,157 | — | — | — | 4,157 |
| **Total** | 61,634 | 42,201 | 15,564 | 8,616 | 4,047 | **132,062** |

### Key dataset findings (preliminary analysis)
- `semantic == "obj"` never co-occurs with `structural == "query"` — "object query" category is structurally impossible in GQA
- No "how many" / counting questions exist in the balanced val split
- ViLT vocabulary covers **96.6%** of questions at question level (≥93% per cell) — fixed vocab is not a structural barrier
- `logical` questions have mean program depth **5.32** vs ≤3.0 for all other structural types

---

## 4. What Has Been Done

### Milestone 1: Full Inference (complete)

**Code:** `milestone1_full_inference/`

| File | Purpose |
|---|---|
| `run_inference.py` | Runs BLIP + ViLT on all 132,062 questions; resumable; reads images from zip |
| `analyze_results.py` | Produces 5×5 matrices, heatmaps, group stats — runs both strict and relaxed evaluation |
| `analysis_blip_mismatch.py` | Identifies and categorises BLIP "correct-ish" failures |
| `predictions/all_predictions.jsonl` | Raw predictions: one JSON row per question, both models, both answers + correctness |
| `results/per_question_stats.csv` | 132,062 rows with strict and relaxed correctness columns for both models |
| `results/strict/` | Full results under standard exact-match normalization |
| `results/relaxed/` | Full results under extended normalization (plural, prep-prefix) |
| `results/blip_mismatch_analysis.csv` | 5,152 BLIP "correct-ish" cases for manual inspection |

---

## 5. Results

### 5.1 Overall Accuracy

| Model | Strict | Relaxed | Δ (relaxed − strict) |
|---|---|---|---|
| BLIP | **52.91%** | **54.12%** | +1.21pp (1,602 questions flipped) |
| ViLT | **50.03%** | **50.96%** | +0.93pp (1,223 questions flipped) |

### 5.2 5×5 Accuracy Matrix (Strict)

**BLIP:**

|  | rel | attr | obj | cat | global |
|---|---:|---:|---:|---:|---:|
| **query** | .382 | .505 | — | .517 | .349 |
| **verify** | .722 | .672 | .786 | — | .956 |
| **logical** | — | .708 | .514 | — | — |
| **choose** | .520 | .640 | — | .729 | .944 |
| **compare** | — | .486 | — | — | — |

**ViLT:**

|  | rel | attr | obj | cat | global |
|---|---:|---:|---:|---:|---:|
| **query** | .368 | .484 | — | .480 | .309 |
| **verify** | .684 | .650 | .740 | — | .951 |
| **logical** | — | .672 | .489 | — | — |
| **choose** | .507 | .537 | — | .607 | .896 |
| **compare** | — | .488 | — | — | — |

**Gap (BLIP − ViLT), positive = BLIP wins:**

|  | rel | attr | obj | cat | global |
|---|---:|---:|---:|---:|---:|
| **query** | +.014 | +.020 | — | +.037 | +.040 |
| **verify** | +.039 | +.022 | +.046 | — | +.005 |
| **logical** | — | +.035 | +.024 | — | — |
| **choose** | +.013 | **+.103** | — | **+.122** | +.049 |
| **compare** | — | **.000** | — | — | — |

### 5.3 Capability Group Accuracy (Strict)

| Group | n | BLIP | ViLT | Gap |
|---|---:|---:|---:|---:|
| S1 Open-ended retrieval (`query`) | 68,167 | .427 | .408 | +.019 |
| S2 Binary perception (`verify`+`logical`) | 43,509 | .660 | .629 | +.031 |
| S3 Constrained choice (`choose`+`compare`) | 20,386 | .589 | .533 | +.056 |
| V1 Relational/spatial (`rel`) | 61,634 | .481 | .461 | +.020 |
| V2 Attribute recognition (`attr`) | 42,201 | .578 | .542 | +.036 |
| V3 Object detection (`obj`) | 15,564 | .564 | .536 | +.028 |
| V4 Categorization (`cat`) | 8,616 | .553 | .501 | +.051 |
| V5 Scene understanding (`global`) | 4,047 | .563 | .529 | +.034 |

### 5.4 Structural and Semantic Marginals (Strict)

**By structural type:**

| Structural | n | BLIP | ViLT | Gap |
|---|---:|---:|---:|---:|
| query | 68,167 | .427 | .408 | +.019 |
| verify | 27,413 | .722 | .688 | +.033 |
| logical | 16,096 | .555 | .528 | +.027 |
| choose | 16,229 | .616 | .545 | +.071 |
| compare | 4,157 | .486 | .488 | -.002 |

**By semantic type:**

| Semantic | n | BLIP | ViLT | Gap |
|---|---:|---:|---:|---:|
| rel | 61,634 | .481 | .461 | +.020 |
| attr | 42,201 | .578 | .542 | +.036 |
| obj | 15,564 | .564 | .536 | +.028 |
| cat | 8,616 | .553 | .501 | +.051 |
| global | 4,047 | .563 | .529 | +.034 |

---

## 6. Key Findings

### Finding 1: BLIP wins in 14/15 cells, but the gap is mostly small and uniform

BLIP outperforms ViLT in 14 of 15 populated cells (all except `compare×attr`, which is essentially tied at ~.487). However, the typical gap is only **+2–4pp**, and it is remarkably uniform across structural types — query (+.019), verify (+.033), logical (+.027) — which is the signature of a generally stronger model rather than an architecturally-motivated pattern.

### Finding 2: The largest BLIP advantage is in `choose`, which contradicts the original hypothesis

The proposal predicted ViLT would be competitive on constrained-choice questions (`choose`, `compare`) because the closed answer space aligns with its classifier. The opposite is observed: `choose` is where BLIP's advantage is **largest** (+.071 avg, with `choose×attr` at +.103 and `choose×cat` at +.122).

**Explanation:** In `choose` questions, both answer candidates appear verbatim in the question text (e.g., "Is it brown or white?"). BLIP's language model decoder naturally conditions on those candidate tokens. ViLT's fixed classifier scores all 3,129 labels ignoring which options are presented, so it may not select either of the two offered answers. This is arguably an architectural effect that survives the pretraining confound.

### Finding 3: `verify×global` is the easiest cell; `query×global` is the hardest

`verify×global` reaches **95.6% (BLIP) / 95.1% (ViLT)** — both models near ceiling. These are simple binary questions about scene-level properties (indoor/outdoor, weather) and always have depth=2 programs.

`query×global` is the hardest: **.349 (BLIP) / .309 (ViLT)**. These are open-ended scene questions ("Which room is it?", "How is the weather?") where the model must produce an exact label from scratch.

### Finding 4: Relational reasoning is the hardest semantic type for both models

V1 (`rel`) has the lowest accuracy for both models (.481 BLIP / .461 ViLT) despite being the largest cell (61,634 questions, 46.7% of the dataset). Spatial and functional relation grounding remains a consistent weakness across architectures.

### Finding 5: `compare×attr` is the only tie — both models struggle equally

Both models score ~.487 on `compare×attr`, the only cell requiring dual-object attribute comparison. Neither architecture handles simultaneous multi-object attention well; the pretraining advantage of BLIP does not help here.

### Finding 6: The comparison is heavily confounded

BLIP and ViLT differ in four compounded dimensions, none of which can be isolated:
1. **Pretraining scale**: ~4.4M vs ~130M pairs (30× difference)
2. **Patch resolution**: ViT-B/32 vs ViT-B/16 (4× more image tokens for BLIP)
3. **Architecture**: encoder-only vs encoder-decoder
4. **Fine-tuning data**: VQA v2 only vs VQA v2 + Visual Genome QA

The uniform BLIP advantage is most consistent with pretraining scale being the dominant factor. The `choose` anomaly (Finding 2) is the clearest candidate for a genuinely architectural effect.

---

## 7. BLIP Evaluation Artefact: "Correct-ish" Failures

Because BLIP generates free-form text rather than selecting a fixed label, exact-match penalises it for surface mismatches that are semantically correct. Among 62,192 BLIP wrong predictions:

| Category | Count | % of BLIP wrong | Acc impact if fixed |
|---|---:|---:|---:|
| `blip_superset` (GT substring of pred) | 3,004 | 4.83% | +2.28pp |
| `blip_subset` (pred substring of GT) | 1,959 | 3.15% | +1.48pp |
| `plural` (singular/plural only) | 1,335 | 2.15% | +1.01pp |
| `prep_prefix` ("on snow" vs "snow") | 281 | 0.45% | +0.21pp |
| `synonym_overlap` | 189 | 0.30% | +0.14pp |
| **Total correct-ish** | **5,152** | **8.28%** | **+3.90pp** |

- **95% of correct-ish cases are in `query`** (open-ended), where BLIP's generative nature produces paraphrases
- `verify`/`logical` have essentially zero correct-ish cases (yes/no answers leave no room for paraphrase)
- If all correct-ish cases were counted as correct, BLIP accuracy would reach **56.8%**

**Current approach:** Both strict (standard exact-match) and relaxed (plural + prep-prefix normalization) evaluation are implemented. The relaxed mode adds +1.21pp for BLIP and +0.93pp for ViLT, preserving the overall rankings and gap structure. Results are saved separately in `results/strict/` and `results/relaxed/`.

---

## 8. What Has Not Been Done Yet

| Analysis | Status | Notes |
|---|---|---|
| Program depth analysis | **Not yet done** | Accuracy vs. depth bin {2,3,4,5,≥6} for both models; may reveal architectural signal that the 5×5 matrix misses |
| OOV breakdown for ViLT | **Not yet done** | Separate ViLT wrong predictions into OOV failures vs. in-vocab errors per cell |
| Error case study (scene graph + attention) | **Not yet done** | Proposed in original proposal; requires attention map extraction |
| Relaxed eval refinement | **Pending** | Inspect `blip_mismatch_analysis.csv` to decide whether `blip_superset` cases should be included in relaxed metric |
| Paper write-up | **Not yet started** | |

---

## 9. Open Questions / Discussion Points

1. **Is the `choose` gap architectural or pretraining?** The +10pp BLIP advantage on `choose×attr` and `choose×cat` is the strongest candidate for an architectural finding. Worth investigating whether BLIP's generation specifically conditions on the candidate words in the question.

2. **What does program depth analysis show?** If BLIP degrades less with depth than ViLT, that would be a genuine architectural signal (generative decoding handles multi-step reasoning better). This is the most important remaining analysis.

3. **How to frame the confound in the paper?** The honest framing is that this is a comparison of two representative pretrained VQA systems, not a controlled ablation. The within-matrix variation (why does `query×rel` score .38 while `verify×global` scores .95?) is the primary finding, regardless of which model wins overall.

4. **Should relaxed eval be the primary metric?** Plural normalization is unambiguous and corrects legitimate evaluation noise. `blip_superset` is more debatable — some cases (GT: `park`, BLIP: `skate park`) may reflect BLIP being *more* correct than GQA's terse ground truth.

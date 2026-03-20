# Evaluation Design Decisions
> Status: draft — will be updated after BLIP mismatch fix
> Last updated: 2026-03-18

This document explains the evaluation issues we encountered and the design
decisions we made. It is intended as a reference for writing the milestone report
and for explaining our methodology to the course instructor.

---

## 1. The ViLT Out-of-Vocabulary (OOV) Problem

### What the problem is

ViLT (`dandelin/vilt-b32-finetuned-vqa`) is a classification model, not a
generative model. At inference time it performs a softmax over a **fixed set of
3,129 answer labels** and outputs the highest-scoring one. It cannot generate
answers outside this set.

These 3,129 labels are the most frequent answer strings from **VQA v2 training
data**. They include both single words (`"dog"`, `"yes"`, `"2"`) and multi-word
phrases (`"living room"`, `"hot dog"`, `"black and white"`).

Because GQA (our evaluation dataset) was created independently from VQA v2, some
valid GQA ground-truth (GT) answers simply do not appear in ViLT's label set. For
those questions, ViLT can never produce the correct answer regardless of how well
it understands the image and question. This is a structural ceiling imposed by the
model architecture, not a reasoning failure.

### Key numbers (before normalization)

| Metric | Value |
|--------|-------|
| ViLT label set size | 3,129 |
| Unique GT answer types in GQA val | 1,469 |
| GT types NOT in ViLT vocab | 449 (30.6% of types) |
| Questions with OOV GT | 4,437 of 132,062 (3.4%) |

The disconnect between 30.6% of *types* and 3.4% of *questions* is important:
OOV answers are long-tail, low-frequency. The most common GQA answers (`"yes"`,
`"no"`, `"left"`, `"right"`, `"man"`, `"woman"`) are all in ViLT's label set.

### Where OOV is concentrated

Binary-answer cells (`verify×*`, `logical×*`) have **zero OOV** because "yes" and
"no" are always in the label set. OOV is concentrated in open-vocabulary cells:

| Cell | OOV% (before norm) |
|------|--------------------|
| query×global | 10.2% |
| query×rel | 7.0% |
| query×cat | 6.8% |
| choose×cat | 3.8% |
| choose×attr | 3.6% |

### Top OOV answer types

The most frequent OOV answers are: `"guy"` (320 questions), `"computer mouse"`
(133), `"material"` (103), `"bookcase"` (83), `"t-shirt"` (75), `"blond"` (71).
These are legitimate, specific answers that GQA uses but VQA v2 apparently does
not include in its top-3,129 vocabulary.

### What the OOV correction actually changes

After applying normalization (rules 1–6) and re-evaluating on the answerable
subset (questions where the normalized GT is in ViLT's normalized label set),
the performance picture is as follows:

| Evaluation subset | n | BLIP | ViLT | Gap |
|---|---|---|---|---|
| Full dataset — strict | 132,062 | 52.91% | 50.03% | +2.88pp |
| Full dataset — normalized | 132,062 | 53.94% | 50.95% | +2.99pp |
| Answerable only — normalized | 128,135 | 55.41% | 52.51% | +2.90pp |
| OOV questions only — normalized | 3,927 | 6.00% | 0.00% | +6.00pp |

The key finding is that **the BLIP–ViLT gap barely changes after OOV exclusion
(2.99pp → 2.89pp, a difference of 0.1pp)**. This tells us two things:

1. **The OOV problem does not meaningfully bias the comparison.** ViLT's lower
   accuracy is not an artifact of being asked questions it structurally cannot
   answer — the gap reflects genuine differences in model capability and
   pretraining.

2. **BLIP's advantage holds on the answerable subset.** BLIP leads ViLT in 14 of
   15 cells even after restricting to questions where ViLT has a valid label. The
   only exception is `compare×attr`, where both models are near random (≈0.49).

3. **OOV questions are hard for BLIP too.** On the 3,998 OOV questions, BLIP
   achieves only 6% despite being a free-form generative model. These questions
   involve long-tail, specific GT answers (e.g., `"t-shirt"`, `"bookcase"`,
   `"blond"`) that BLIP does not generate exactly — likely predicting close
   synonyms that fail exact-match. This hints at the BLIP mismatch problem
   discussed in Section 2.

**Summary sentence for the report:** We identified that 3.4% of GQA questions
have GT answers outside ViLT's fixed label set (a structural ceiling, not a
reasoning failure). After applying string normalization (rules 1–6) and
re-evaluating on the answerable subset, the BLIP–ViLT performance gap changes by
only 0.1pp (2.99pp → 2.89pp), confirming that the OOV constraint does not
materially bias our comparison. We report all three evaluation modes (strict,
normalized, normalized-OOV-excluded) for full transparency.

---

## 2. The BLIP Mismatch Problem

### What the problem is

BLIP (`Salesforce/blip-vqa-base`) is a generative model — it produces free-form
text answers. This creates a different kind of mismatch with GQA's single GT
answer: the model may express the correct answer in a different surface form.
BLIP is structurally more exposed to this problem than ViLT, which always outputs
one label from a predefined list and therefore has a consistent surface form.

### Investigation: categorising every BLIP failure

We ran a mismatch classifier (`analysis_blip_mismatch.py`) over all 62,192
BLIP-wrong cases, tagging each with one of six surface-form patterns:

| Tag | Definition | Example |
|-----|-----------|---------|
| `plural` | differ only by trailing -s / -es | GT=`brick` BLIP=`bricks` |
| `blip_superset` | GT is a substring of BLIP's output | GT=`blond` BLIP=`blonde` |
| `blip_subset` | BLIP's output is a substring of GT | GT=`dark blue` BLIP=`blue` |
| `article_only` | differ only by leading a/an/the | GT=`street` BLIP=`the street` |
| `prep_prefix` | BLIP adds a prepositional prefix | GT=`snow` BLIP=`on snow` |
| `synonym_overlap` | share a content word, neither contains the other | — |

The script found 5,152 "correct-ish" cases (8.3% of BLIP-wrong). However, on
closer inspection, many of these are **false positives from the substring
detector**, not genuine mismatches.

### What we found per structural type

**`verify` — 0 correct-ish.** Answers are always yes/no; no surface variation
possible. Nothing to address.

**`logical` — 16 flagged, all false positives or genuinely wrong.**
The short-word substring problem: `"no"` appears inside `"snow"`, `"notes"`,
`"snowboards"` as a bare string, triggering the superset flag even though BLIP
predicted a completely unrelated word. Every one of these 16 cases is a detector
artefact; BLIP genuinely got the answer wrong.

**`compare` — 20 flagged, mostly false positives.**
The most common pattern: GT=`"woman"`, BLIP=`"man"` — flagged because `"man" ⊂
"woMAN"` as a substring. BLIP predicted the wrong person entirely; the detector
misfired. A handful of superset cases (GT=`"color"`, BLIP=`"they are all same
color"`) are borderline but we keep them as wrong since GQA expects a one-word
answer.

**`choose` — 223 flagged, all treated as wrong (by design).**
`choose` questions explicitly offer two options in the question text ("wooden or
metallic?"). Even morphological variants like GT=`"wooden"` BLIP=`"wood"` are
counted as wrong: the model should have selected from the given options, not
generated its own form. We deliberately apply no leniency to this category.

**`query` — 4,893 flagged, the only category worth examining.**

Decomposed by actual pattern:

| Pattern | Approx. count | Verdict |
|---------|--------------|---------|
| Plural/singular: GT=`brick` BLIP=`bricks` | ~1,330 | ✅ Already fixed by rule 6 |
| Spelling: GT=`blond` BLIP=`blonde` | ~150 | ✅ Fixed: add `blonde→blond` to map |
| BLIP over-specifies colour: GT=`blue` BLIP=`blue and white` | ~600 | ❌ Genuinely wrong |
| BLIP drops modifier: GT=`dark blue` BLIP=`blue` | ~700 | ❌ Genuinely wrong |
| Short-word substring accidents (`no`/`snow`) | small | ❌ Detector false positive |

The **colour over-specification** pattern (BLIP=`"blue and white"` when GT=`"blue"`)
is a systematic BLIP behaviour: it describes the full visual pattern of a
striped or multi-coloured object rather than the single annotated colour. This is
a genuine model limitation, not a surface mismatch, and should remain wrong.

The **modifier-drop** pattern (BLIP=`"blue"` for GT=`"dark blue"`) reflects
BLIP's difficulty with fine-grained colour precision. Also a genuine error.

### What the normalization already fixes

After applying rules 1–6 to both GT and prediction:
- **Plural/singular** (~1,330 query cases): fixed by rule 6 (depluralize)
- **`blonde → blond`** (~150 cases): fixed by adding to the rule-5 spelling map
- **Articles, number words**: fixed by rules 3–4 (were already in strict baseline)

### What the normalization does NOT fix (and we accept)

- **Colour over-specification** (GT=`blue`, BLIP=`blue and white`): keeping as
  wrong. This is a real reasoning limitation, not annotation noise.
- **Modifier drop** (GT=`dark blue`, BLIP=`blue`): keeping as wrong.
- **Detector false positives** in `logical`/`compare`: the underlying predictions
  are genuinely wrong; no correction needed.

### Conclusion and accuracy impact

After applying normalization rules 1–6 (including the `blonde→blond` fix),
the residual uncorrected mismatch cases account for **less than 0.5pp** of BLIP
accuracy. The mismatch problem is real but small; the normalization pipeline
handles the defensible cases, and the remaining failures reflect genuine model
limitations rather than annotation artefacts.

**Concrete numbers (final, after all fixes):**

| | BLIP | ViLT | Gap |
|---|---|---|---|
| Strict exact-match | 52.91% | 50.03% | +2.88pp |
| Normalized (rules 1–6 incl. blonde fix) | 53.94% | 50.95% | +2.99pp |
| Normalization lift | **+1.03pp** | **+0.92pp** | — |

The normalization lifted BLIP by 1.03pp and ViLT by 0.92pp — nearly equal gains,
confirming the rules are symmetric and do not favour either model. Of BLIP's
1.03pp lift, the `blonde→blond` fix alone accounts for ~0.03pp (40 questions);
the remainder is plural/singular and article/number normalization.

**Summary sentence for the report:** We investigated BLIP's generative outputs
for surface-form mismatches using a pattern classifier across 62,192 wrong
predictions. We found that the largest legitimate mismatch categories
(plural/singular and the `blonde/blond` spelling variant) are fully addressed by
our normalization rules 1–6. The remaining flagged cases are either detector
false positives or genuine reasoning errors (colour over-specification, modifier
drop) that should remain wrong. After normalization, BLIP gains 1.03pp and ViLT
gains 0.92pp — symmetric improvements that do not alter the relative comparison.
The residual unfixed mismatch accounts for less than 0.5pp and does not
materially affect our conclusions.

---

## 3. Why Simple String Normalization (Rules 1–6)

### What we normalize

We apply the following rules to **both** the GT answer and the model prediction
before comparing them. Rules are applied in order:

| Rule | Operation | Example |
|------|-----------|---------|
| 1 | Lowercase + strip whitespace | `"Street"` → `"street"` |
| 2 | Strip leading/trailing punctuation | `"yes."` → `"yes"` |
| 3 | Strip leading article (a/an/the) | `"the street"` → `"street"` |
| 4 | Number words → digits | `"two"` → `"2"` |
| 5 | British → American spelling (word-level) | `"grey"` → `"gray"` |
| 6 | Plural → singular (conservative) | `"chairs"` → `"chair"` |

### Why normalization is applied to BOTH sides symmetrically

Normalization is a pre-processing step on the **dataset**, not a model-specific
correction. We canonicalize the surface form of both the GT answer and the model
prediction before comparing them. Both models benefit equally from this, and the
comparison remains fair.

Applied symmetrically, this also allows us to check ViLT OOV against the
*normalized* label set rather than the raw one, which slightly reduces the OOV
count (e.g., GT `"grey"` → normalized `"gray"`, which IS in ViLT's label set).

### Why we chose rules 1–6 specifically

**Rules 1–4** are the conventional VQA evaluation normalization, used by the
original VQA v2 paper and most subsequent work. Applying these rules is standard
practice.

**Rule 5 (British spelling)** addresses a systematic annotation artifact: GQA was
annotated and may include British spellings (`"grey"`, `"colour"`) while ViLT's
VQA v2 vocabulary uses American spellings. This is an annotation inconsistency,
not a model failure.

**Rule 6 (plural → singular)** is conservative — we only strip unambiguous plural
suffixes and protect common exceptions (`"yes"`, `"bus"`, etc.). This addresses
cases where GQA and a model use different number agreements for the same object.

### What we deliberately exclude

We do **not** apply:
- Preposition stripping ("`on table`" → "`table`"): too aggressive, changes meaning
- Synonym expansion ("`woman`" ↔ "`lady`"): changes the scope of what "correct" means
- Substring/partial matching ("`blue`" counts for "`dark blue`"): asymmetric benefit,
  hard to bound, and methodologically harder to defend

Synonym expansion and partial matching are collectively called **semantic
relaxation** — they would further close the BLIP–ViLT gap and would benefit
BLIP's generative output more than ViLT's classifier output, making the
comparison less interpretable. We leave these for future work.

---

## 4. Three Evaluation Modes and What Each Reports

### Mode 1: Strict

- **Normalization**: rules 1, 3, 4 (conventional VQA baseline: lowercase, strip
  article, number words→digits)
- **Subset**: all 132,062 questions
- **Reports**: BLIP accuracy, ViLT accuracy, gap
- **Purpose**: replicates the conventional VQA evaluation baseline so our results
  are comparable to prior work

### Mode 2: Normalized

- **Normalization**: rules 1–6 (all rules including British spelling + depluralize)
- **Subset**: all 132,062 questions
- **Reports**: BLIP accuracy, ViLT accuracy, gap
- **Purpose**: primary evaluation — reduces surface-form artifacts symmetrically
  for both models. The gain from strict → normalized measures how much both models
  were penalized by surface-form mismatches.

### Mode 3: Normalized-OOV-Excluded

- **Normalization**: rules 1–6 (same as normalized)
- **Subset**: questions where normalized GT is in ViLT's normalized label set
  (~97.0% of questions after normalization)
- **Reports**: BLIP accuracy, ViLT accuracy, gap (both models on the same subset)
- **Purpose**: upper-bound comparison — removes questions ViLT is structurally
  unable to answer. This isolates reasoning ability from vocabulary coverage.
  The gap between Mode 2 and Mode 3 for ViLT = cost of the fixed vocabulary.

### What to report in the paper

| Result | Table | Heatmap |
|--------|-------|---------|
| Strict accuracy | ✓ (summary row) | ✓ |
| Normalized accuracy | ✓ (primary) | ✓ |
| Normalized-OOV-excluded (ViLT) | ✓ | ✓ |
| Strict → Normalized lift | ✓ (delta column) | — |
| Per-cell OOV rate | ✓ | ✓ (separate) |

The **5-panel heatmap** (BLIP strict, BLIP norm, ViLT strict, ViLT norm, ViLT
OOV-excluded) is the key figure for the report — it shows the full picture in one
image.

---

## 5. Why This Is Still a Fair Comparison

A potential reviewer concern: "The comparison is unfair because BLIP and ViLT have
different pretraining objectives and answer mechanisms."

Our response (to frame in the report):

1. **We acknowledge the asymmetry explicitly.** BLIP is a generative encoder-
   decoder; ViLT is a fixed-label classifier. They cannot be perfectly equalized.

2. **Our contribution is not "BLIP beats ViLT."** The contribution is that
   *per-type accuracy varies substantially in ways that aggregate accuracy hides*,
   and that this variation is systematic and interpretable. The model comparison is
   a vehicle for demonstrating this type-aware insight, not the end goal.

3. **The normalized-OOV-excluded mode levels the vocabulary playing field** by
   restricting evaluation to questions both models can theoretically answer.
   Within that subset, the comparison is as fair as the architectures allow.

4. **Both models are evaluated on the same questions** in every mode. We never
   apply a rule that benefits one model but not the other.

---

## 6. Open Questions (to revisit after BLIP analysis)

- [ ] Does BLIP generate longer answers than ViLT? If so, do plural/article rules
  help BLIP more? Quantify the lift per model.
- [ ] Are there BLIP-specific surface patterns not covered by rules 1–6? (e.g.,
  word order, paraphrasing). If yes, consider a small synonym map.
- [ ] Should we report a Mode 4: normalized-OOV-excluded with only BLIP? (i.e., OOV
  for ViLT but not for BLIP, evaluated separately). Decision: probably not needed
  since BLIP has no OOV by construction (it generates freely).

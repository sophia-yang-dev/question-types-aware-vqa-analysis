# Evaluation Design: Question-Type-Aware VQA Analysis
**Project:** Comparing BLIP and ViLT on GQA via a Structured Question Taxonomy
**Dataset:** GQA Balanced Validation Split (132,062 questions)

---

## Background: How GQA Questions Are Generated

GQA questions are not crowd-sourced — they are **automatically generated from scene graphs** using a compositional grammar. Each scene graph encodes objects, their attributes, and their spatial/functional relationships. The generation program instantiates question templates by traversing these graphs and filling in slots, producing a question + ground-truth answer pair that is guaranteed to be answerable from the image.

Every generated question is tagged along **two independent axes**:

- **Structural type** — the *syntactic/logical form* of the question (what kind of reasoning step it demands)
- **Semantic type** — the *content target* of the question (what aspect of the visual scene it is about)

This two-axis scheme gives GQA a principled, non-overlapping taxonomy that is ideal for systematic model evaluation. Our study exploits it directly.

---

## Axis 1: Structural Types (The Reasoning Form)

The structural axis describes *what the model must do* to answer the question correctly. It is independent of what the question is about.

### `query` — Open-ended retrieval
The model must retrieve a specific answer from an open vocabulary. There is no constraint on the answer space — it could be a color, a name, a material, a spatial direction, a category label, or anything else.

> *"What is the color of the shirt?"* → `blue`
> *"What kind of vehicle is to the right of the man?"* → `suv`
> *"Which room is this?"* → `bathroom`

**Capability tested:** Free-form visual recognition + answer generation. The model must both correctly identify the relevant image region and produce the right label string.

### `verify` — Binary yes/no judgment
The model is presented with a proposition about the image and must confirm or deny it. The answer is always `yes` or `no`.

> *"Is the tall man wearing a helmet?"* → `no`
> *"Do the flowers have yellow color?"* → `no`
> *"Is there a towel in this scene?"* → `yes`

**Capability tested:** Perceptual verification. The model must locate the referenced entity/attribute and judge whether the stated property holds. There is no retrieval burden — the model only needs to say yes or no.

### `logical` — Boolean combination
The model must answer a compound yes/no question that involves a Boolean operation (AND, OR, NOT) over two or more scene facts. The answer is always `yes` or `no`.

> *"Are there white pillows or kites?"* → `yes`
> *"Are there both doors and windows?"* → `no`
> *"Does the bowl look white and small?"* → `yes`

**Capability tested:** Multi-step reasoning. The model must evaluate multiple sub-propositions independently and then combine them with a logical operator. This is structurally harder than `verify` because getting the logic wrong (e.g., treating OR as AND) produces a wrong answer even if the perceptual recognition is correct.

### `choose` — Forced binary choice
The model is given exactly two candidate answers (A or B) embedded in the question, and must pick the correct one. The answer is always one of the two offered options.

> *"On which side is the cup, the left or the right?"* → `right`
> *"Which color is the castle, brown or white?"* → `brown`
> *"Is it an indoors or outdoors picture?"* → `indoors`

**Capability tested:** Discrimination under constrained answer space. Unlike `query`, the model does not need to retrieve an answer from scratch — it only needs to evaluate two candidates and select the better one. This reduces the difficulty of answer generation but still requires accurate perception.

### `compare` — Cross-object attribute comparison
The model must compare the same attribute (e.g., color, size, material) across two different objects and judge whether they are the same or different. The answer is almost always `yes` or `no`.

> *"Are the sky and the shirt the same color?"* → `yes`
> *"Is the color of the trash can different than the boot?"* → `no`
> *"Do the net and the t-shirt have different colors?"* → `yes`

**Capability tested:** Relational attribute reasoning. The model must attend to two separate image regions, extract the same attribute from each, and compare them — a compositional operation that requires simultaneous multi-object grounding.

---

## Axis 2: Semantic Types (The Visual Content Target)

The semantic axis describes *what part of the scene* the question is about. It is independent of how the question is structured.

### `rel` — Spatial and functional relations
The question is about the **relationship between objects** — their spatial arrangement ("to the left of", "in front of", "on top of") or functional role ("wearing", "holding", "riding").

> *"What kind of vehicle is to the right of the man wearing a coat?"* → `suv`
> *"Is there any lettuce on the plate?"* → `no`
> *"Is the kid to the left or to the right of the umbrella?"* → `right`

**Visual skill required:** Spatial language understanding + object relationship grounding. The model must parse relational phrases and spatially ground two (or more) objects relative to each other.

### `attr` — Object attributes
The question is about a **property of a specific object** — its color, material, texture, size, shape, position within the frame, or state.

> *"What is the trash can made of?"* → `metal`
> *"Do the flowers have yellow color?"* → `no`
> *"Which color is the castle, brown or white?"* → `brown`

**Visual skill required:** Fine-grained visual feature recognition. The model must localize a specific object and extract a low-level or mid-level visual property from it.

### `obj` — Object existence and identity
The question is about **whether an object exists** in the scene, or asks to identify objects with certain properties. This semantic type only occurs under `verify` and `logical` (never under `query`), because object existence is inherently a yes/no question.

> *"Is there a towel in this scene?"* → `yes`
> *"Are there white pillows or kites?"* → `yes`
> *"Do you see phones that are not pink?"* → `yes`

**Visual skill required:** Object detection and enumeration. The model must detect whether one or more specified objects are present, which is a core visual recognition task.

### `cat` — Object category / semantic class
The question asks for the **category name or type** of an object, testing whether the model can label it at the correct semantic level of abstraction (e.g., "what kind of furniture?" → `couch`).

> *"What is the name of this piece of furniture?"* → `couch`
> *"Which kind of device is in this picture?"* → `screen`
> *"What type of appliance is not black, the stove or the cooker?"* → `stove`

**Visual skill required:** Semantic categorization. The model must recognize not just that an object exists, but map it to the right node in an ontology — which requires broader visual-semantic alignment than attribute recognition.

### `global` — Scene-level understanding
The question is about **holistic scene properties** that require understanding the entire image rather than a specific object — room type, location, weather, time of day, indoor/outdoor context.

> *"Which room is it?"* → `bathroom`
> *"How is the weather?"* → `overcast`
> *"Is it an indoors or outdoors picture?"* → `indoors`

**Visual skill required:** Scene-level comprehension. The model must integrate evidence from the entire image rather than attending to a specific object or region.

---

## The 5×5 Evaluation Matrix

Crossing the two axes gives a 5×5 grid. Not all 25 cells are populated — GQA's generation grammar produces only **15 non-empty combinations**. The 10 empty cells are structural zeros (those question forms do not arise from the generation logic, e.g., you cannot `compare` object existence).

### Question Counts

|  | **rel** | **attr** | **obj** | **cat** | **global** | **Row total** |
|---|---:|---:|---:|---:|---:|---:|
| **query** | 40,278 | 18,092 | — | 7,185 | 2,612 | **68,167** |
| **verify** | 15,602 | 8,053 | 2,879 | — | 879 | **27,413** |
| **logical** | — | 3,411 | 12,685 | — | — | **16,096** |
| **choose** | 5,754 | 8,488 | — | 1,431 | 556 | **16,229** |
| **compare** | — | 4,157 | — | — | — | **4,157** |
| **Col total** | **61,634** | **42,201** | **15,564** | **8,616** | **4,047** | **132,062** |

### Percentage of Full Dataset

|  | **rel** | **attr** | **obj** | **cat** | **global** |
|---|---:|---:|---:|---:|---:|
| **query** | 30.5% | 13.7% | — | 5.4% | 2.0% |
| **verify** | 11.8% | 6.1% | 2.2% | — | 0.7% |
| **logical** | — | 2.6% | 9.6% | — | — |
| **choose** | 4.4% | 6.4% | — | 1.1% | 0.4% |
| **compare** | — | 3.1% | — | — | — |

The dataset is heavily skewed toward `query×rel` (30.5%) and `query×attr` (13.7%) — together these two cells account for 44% of all questions. The smallest populated cells are `choose×global` (556, 0.4%) and `verify×global` (879, 0.7%).

### What We Will Measure in Each Cell

For each of the 15 cells, we report:
- **Exact-match accuracy** for BLIP and ViLT (after normalization: strip, lowercase, number words → digits, remove leading articles)
- **Accuracy gap** (BLIP − ViLT), signed so that positive = BLIP wins
- These results form a **5×5 heatmap** (two heatmaps: one per model; one heatmap of the gap)

This matrix is the primary result of the paper.

---

## Capability-Based Groupings (Secondary Analysis)

The 15 cells can be meaningfully grouped along either axis to tell a higher-level story about model capabilities. We use these groupings in the discussion section to synthesize the matrix findings.

### Grouping by Structural Axis: What Kind of Reasoning?

#### Group S1: Open-ended retrieval (`query` row)
**Cells:** query×rel, query×attr, query×cat, query×global
**n = 68,167 (51.6%)**

The model must produce an answer without any scaffolding. This is the most demanding structural form because the answer could be anything — a color, a name, a location, a category. It directly exposes the gap between BLIP (generative, can produce arbitrary strings) and ViLT (classifier, limited to 3,129 labels).

*Expected pattern:* BLIP should have an architectural advantage here, but may be penalized by exact-match evaluation when it paraphrases a correct answer (e.g., "automobile" vs. "car").

#### Group S2: Binary perception (`verify` + `logical` rows)
**Cells:** verify×rel, verify×attr, verify×obj, verify×global, logical×obj, logical×attr
**n = 43,509 (32.9%)**

All answers are `yes` or `no`. The model's task is to confirm or deny a proposition. `logical` adds the extra challenge of Boolean combination. Both models have `yes`/`no` in their answer space, so this is a level playing field architecturally. Performance differences here reflect differences in *perceptual accuracy* and *reasoning ability*, not answer-space coverage.

*Expected pattern:* Both models should perform comparably on `verify`; `logical` may expose a bigger gap if one model is better at multi-step reasoning. Watch for a high baseline from majority-class prediction (GQA is balanced 50/50 for yes/no).

#### Group S3: Constrained choice (`choose` + `compare` rows)
**Cells:** choose×rel, choose×attr, choose×cat, choose×global, compare×attr
**n = 20,386 (15.4%)**

The answer space is partially revealed in the question itself. For `choose`, the two options are explicit. For `compare`, the answer is almost always yes/no. This reduces the retrieval burden — the model can focus on discrimination rather than generation.

*Expected pattern:* ViLT may perform relatively better here than in Group S1 because the constrained answer space aligns with its classification paradigm. BLIP may still win if it understands the option framing in the question text.

---

### Grouping by Semantic Axis: What Visual Skill Is Needed?

#### Group V1: Relational/spatial reasoning (`rel` column)
**Cells:** query×rel, verify×rel, choose×rel
**n = 61,634 (46.7%)**

All questions require the model to understand spatial language ("to the left of", "on top of") or functional relations ("wearing", "holding"). This is the hardest visual grounding task because it requires simultaneously localizing multiple objects and parsing their geometric or semantic relationship.

*Expected pattern:* Both models likely struggle here relative to other semantic types. BLIP's cross-attention over image patches may give it an edge in spatial reasoning.

#### Group V2: Attribute recognition (`attr` column)
**Cells:** query×attr, verify×attr, logical×attr, choose×attr, compare×attr
**n = 42,201 (32.0%)**

All questions target a specific object attribute: color, material, texture, position, size. These are fine-grained visual features that require careful localization + feature extraction.

*Expected pattern:* Both models should perform well on color/position (common in training data). Material and texture may be harder. `compare×attr` is uniquely demanding because it requires extracting the same attribute from two different objects.

#### Group V3: Object detection (`obj` column)
**Cells:** verify×obj, logical×obj
**n = 15,564 (11.8%)**

All questions ask whether one or more objects exist. This maps closely to standard object detection capability. Answers are always yes/no.

*Expected pattern:* Both models are trained on VQA-style data with heavy object existence questions. Expect high accuracy for both, with differences reflecting training data overlap with GQA object vocabulary.

#### Group V4: Categorization (`cat` column)
**Cells:** query×cat, choose×cat
**n = 8,616 (6.5%)**

The model must name or select the correct semantic category for an object ("what kind of furniture?", "what type of appliance?"). This requires broad visual-semantic knowledge of object ontologies.

*Expected pattern:* BLIP may have better category generalization due to larger-scale pretraining. ViLT is limited to category names that appear in VQA v2.0's label set.

#### Group V5: Scene understanding (`global` column)
**Cells:** query×global, verify×global, choose×global
**n = 4,047 (3.1%)**

The model must understand the entire scene holistically — room type, location, weather, indoor/outdoor context. No specific object needs to be localized.

*Expected pattern:* Smallest group, so results may be noisy. Both models should handle indoor/outdoor well (very common in VQA training). Weather and specific room types may be harder.

---

## Connection to BLIP vs. ViLT Architectures

Understanding the architectural differences between BLIP and ViLT is essential for interpreting the matrix results. The two models represent fundamentally different paradigms for VQA.

### ViLT: Encoder-only Classifier

ViLT encodes the concatenated image patches + question tokens with a single Transformer encoder, then applies a linear classifier head over a **fixed vocabulary of 3,129 labels** derived from VQA v2.0 training. The answer is always the top-scoring label.

| Property | Implication for our evaluation |
|---|---|
| Fixed label set | Cannot produce answers outside its 3,129 vocabulary; 3.4% of GQA questions have OOV ground-truth answers |
| Classification, not generation | Answer is always a single label — no multi-word answers, no paraphrasing |
| Lightweight image encoding | Processes image as flat patch sequence with no region proposals; may struggle with fine-grained spatial reasoning |
| Fast inference | ~0.063s/sample on CPU; ~4× faster than BLIP |
| Closed answer space | Naturally well-suited to `verify`/`choose`/`compare` (yes/no or binary options) |

**ViLT is expected to be strongest on:** verify×*, choose×*, compare×attr — any structural type where the correct answer is a common VQA label (`yes`, `no`, `left`, `right`, color words).

**ViLT is expected to struggle on:** query×cat, query×global, query×rel — open-ended questions where the answer is a specific object name or scene descriptor that may not appear in VQA v2.0 training labels.

### BLIP: Generative Encoder-Decoder

BLIP uses a vision encoder (ViT) to produce image features, which are fed to a language model that **generates the answer token by token** via autoregressive decoding. The answer is unconstrained — it can be any string.

| Property | Implication for our evaluation |
|---|---|
| Generative answer | Can produce any string; not limited to a fixed vocabulary |
| Larger-scale pretraining | Pretrained on hundreds of millions of image-text pairs; richer visual-semantic alignment |
| Autoregressive decoding | Slower (~0.266s/sample); higher variance due to generation dynamics |
| Multi-word answers possible | May generate "a dog" when GQA says "dog" — penalized by exact-match even if semantically correct |
| No explicit answer space constraint | Evaluated fairly on open-ended cells; may be under-penalized on closed cells (generates `yes` or `no` but may also generate longer strings) |

**BLIP is expected to be strongest on:** query×rel, query×attr, query×cat, query×global — open-ended questions where generative decoding allows it to produce the exact right string.

**BLIP is expected to struggle on:** compare×attr — requires simultaneously attending to two objects; generative models may hallucinate or anchor on just one.

### Predicted Accuracy Pattern in the Matrix

This table gives our prior hypothesis for who will win in each cell, *before* running the full evaluation. After running, we can compare actual results to these predictions and discuss surprises.

|  | **rel** | **attr** | **obj** | **cat** | **global** |
|---|:---:|:---:|:---:|:---:|:---:|
| **query** | BLIP↑ | BLIP↑ | — | BLIP↑ | BLIP↑ |
| **verify** | Tie | Tie | Tie | — | Tie |
| **logical** | — | ViLT? | ViLT? | — | — |
| **choose** | Tie | Tie | — | Tie | Tie |
| **compare** | — | ViLT? | — | — | — |

*Legend: BLIP↑ = BLIP predicted to win; ViLT? = ViLT predicted to have slight edge; Tie = expected to be close.*

The most interesting cells are the **`logical` and `compare` rows** — these require multi-step reasoning that neither model is explicitly trained for, so the winner will be the model whose pretraining best incidentally develops that capability.

---

## Evaluation Protocol

### Metric
**Exact-match accuracy** after normalization:
```python
import re
NUM_MAP = {'zero':'0','one':'1','two':'2','three':'3','four':'4',
           'five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}

def normalize(s):
    s = s.strip().lower()
    if s in NUM_MAP:
        s = NUM_MAP[s]
    s = re.sub(r'^(a |an |the )', '', s)
    return s

correct = normalize(predicted) == normalize(ground_truth)
```

### Data Scope
Given CPU runtime constraints (~9.8 hrs for BLIP over all 132K questions), we will use a **stratified random sample of ~10,000 questions**, sampling proportionally from each of the 15 populated cells. This ensures every cell has enough examples for reliable accuracy estimation while keeping total runtime under 1 hour for BLIP.

Minimum sample per cell: ~100 questions (even the smallest cell, `choose×global` with 556 total, yields a reasonable sample). Exact per-cell sample sizes will be computed as `min(n_cell, round(10000 × n_cell / 132062))`.

### Reporting
1. **Primary table:** 5×5 accuracy matrix for each model + a 5×5 gap matrix (BLIP − ViLT)
2. **Heatmaps:** Visual 5×5 heatmaps of accuracy and gap (matplotlib/seaborn)
3. **Marginal analysis:** Row means (structural type) and column means (semantic type) to identify the strongest structural/semantic effects
4. **Capability group summary:** Accuracy aggregated per group (S1/S2/S3 and V1/V2/V3/V4/V5) with confidence intervals

---

## Summary: Why This Design is Principled

1. **Complete coverage:** The 15 cells cover 100% of the GQA balanced validation set with no overlap and no arbitrary exclusions.
2. **Grounded in GQA's own taxonomy:** We are not inventing categories — we are using the dataset's native annotation directly, which makes comparisons reproducible and interpretable.
3. **Two levels of analysis:** The 5×5 matrix gives fine-grained results; the capability groupings give interpretable high-level conclusions. Both levels inform each other.
4. **Architecturally motivated predictions:** The BLIP vs. ViLT distinction maps cleanly onto the structural axis (open-ended vs. closed answer space), making our evaluation design directly relevant to the models' design choices.
5. **Scalable to future work:** The same matrix design can be applied to other VQA models or other datasets with two-axis taxonomies.

---

## Error Analysis: Reasoning Chain Depth

Beyond the 5×5 accuracy matrix, we propose a secondary error analysis that cuts across all cells using a finer-grained signal already present in the GQA annotations: **functional program depth**.

### What GQA Functional Programs Are

Every question in the GQA dataset is annotated with a `semantic` field containing a **functional program** — a small DAG (directed acyclic graph) of reasoning operations that was used to generate the question from the scene graph. This program is not something we have to infer; it is stored directly in `val_balanced_questions.json` alongside each question.

Each program is a list of steps, where each step has:
- **`operation`**: the reasoning primitive being applied (e.g., `select`, `relate`, `query`, `exist`, `filter color`, `and`, `or`, `same color`, …)
- **`dependencies`**: indices of earlier steps this step takes as input (encodes the DAG edges)
- **`argument`**: the object name, relation type, or attribute being operated on, with the resolved scene graph object ID in parentheses

**Example — a depth-2 question** (`query × attr`):
```
Q: "What color is the shirt?"
Program:
  [0] select: shirt (3827902)          ← locate the shirt in the scene graph
  [1] query: color  [depends on 0]     ← read its color attribute
```
Depth = 2 steps. The model needs to locate one object and read one attribute.

**Example — a depth-3 question** (`query × rel`):
```
Q: "Who is the fence in front of?"
Program:
  [0] select: fence (1248687)                        ← locate the fence
  [1] relate: person, in front of, o (1248689) [0]   ← follow the relation to find the person
  [2] query: name  [depends on 1]                    ← read the name of that person
```
Depth = 3 steps. The model must locate an object, traverse a relation, then retrieve a property.

**Example — a depth-3 chain question** (`choose × rel`):
```
Q: "Is the lady to the right or left of the boy that is to the left of the giraffe?"
Program:
  [0] select: giraffe (1248688)
  [1] relate: boy, to the left of, s (3827903)  [0]   ← find boy relative to giraffe
  [2] choose rel: lady, to the left of|to the right of, s (3827887)  [1]  ← choose lady's side relative to boy
```
Depth = 3 steps, but involves two chained spatial relations.

**Example — a depth-5+ question** (`logical × obj`):
```
Q: "Are there both zebras and giraffes in the photo?"
Program:
  [0] select: zebra (…)
  [1] exist  [0]          ← does zebra exist?
  [2] select: giraffe (…)
  [3] exist  [2]          ← does giraffe exist?
  [4] and  [1, 3]         ← combine both existence checks
```
Depth = 5 steps. The model must run two independent sub-queries and combine them logically.

### Program Depth Distribution in the Dataset

Analyzing all 132,062 questions:

| Depth | Count | % of dataset |
|---:|---:|---:|
| 2 | 36,743 | 27.8% |
| 3 | 62,810 | 47.6% |
| 4 | 17,875 | 13.5% |
| 5 | 11,036 | 8.4% |
| 6 | 85 | 0.1% |
| 7 | 3,506 | 2.7% |
| 8–9 | 7 | ~0.0% |

The vast majority of questions are depth 2–3 (75.4%). Depth 4–5 account for ~22%. Depth 7 questions (2.7%) are an interesting spike — these are almost exclusively `logical × obj` questions with nested OR/AND over filtered object sets.

### Mean Depth by Structural Type

This is the key insight: **depth is not uniformly distributed across the 5×5 matrix**. It varies strongly by structural type:

| Structural type | Mean depth | Depth distribution |
|---|---:|---|
| choose | 2.49 | mostly depth 2–3 |
| verify | 2.72 | depth 2–4 |
| compare | 2.89 | depth 2–3 |
| query | 2.95 | depth 2–4 |
| **logical** | **5.32** | depth 4–9; driven by AND/OR sub-programs |

`logical` is dramatically deeper than all other types — mean depth 5.32 vs. ≤3.0 for everything else. This is because logical questions require running two independent sub-programs and combining their results, effectively doubling the program length.

By semantic type:

| Semantic type | Mean depth |
|---|---:|
| global | 2.00 (always depth 2) |
| attr | 2.59 |
| cat | 2.72 |
| obj | 4.98 (deep due to logical×obj) |
| rel | 3.17 |

`global` is always exactly depth 2 (`select → query`), meaning it is the structurally simplest semantic type. `obj` has high mean depth because it co-occurs almost exclusively with `logical`, which has deep programs.

### How to Compute Depth in Code

Program depth is trivially readable from the existing annotation — no graph traversal required:

```python
def program_depth(question: dict) -> int:
    """Number of operations in the functional program. Directly from annotation."""
    return len(question['semantic'])

# Example usage
for qid, q in data.items():
    q['program_depth'] = program_depth(q)
```

You can also extract additional features from the program for finer analysis:

```python
def program_features(question: dict) -> dict:
    program = question['semantic']
    ops = [step['operation'] for step in program]
    return {
        'depth': len(program),
        'n_relate': ops.count('relate'),          # number of relation traversals
        'n_filter': sum(1 for op in ops if op.startswith('filter')),  # attribute filters
        'has_logical': any(op in ('and', 'or') for op in ops),        # boolean combination
        'has_comparison': any(op.startswith(('same', 'different', 'common', 'choose older',
                                              'choose younger', 'choose healthier')) for op in ops),
        'n_selects': ops.count('select'),         # number of anchor objects
    }
```

### Proposed Error Analysis: Accuracy vs. Program Depth

**Primary analysis:** For each model (BLIP, ViLT), plot **exact-match accuracy as a function of program depth** (depth 2 through 7), both overall and broken down by structural type. This answers:

> *Does model accuracy decrease as questions require more reasoning steps? And does the rate of degradation differ between BLIP and ViLT?*

**Procedure:**
1. Run inference on the stratified 10K sample (see Evaluation Protocol above)
2. For each question, look up `len(q['semantic'])` as the depth label — no extra computation needed
3. Group results by depth bin: {2, 3, 4, 5, ≥6}
4. Compute accuracy per bin per model
5. Plot as a line chart (depth on x-axis, accuracy on y-axis, one line per model)

**Secondary breakdown:** Within each structural type, plot accuracy vs. depth. This isolates the depth effect from the type effect. For example:
- Within `query` (depth 2–4): does accuracy drop from depth-2 to depth-4 queries?
- Within `logical` (depth 4–9): does accuracy drop further at depth 7 vs. depth 5?

**Tertiary analysis — operation-type breakdown:** Since each program step has a named operation, you can flag questions by which operations they contain and compute accuracy per operation type. For instance:
- Questions containing `relate` ops: accuracy of models that must traverse spatial relations
- Questions containing `filter color` then `relate`: accuracy on color-filtered relational queries
- Questions containing `and` or `or`: accuracy on boolean-combined questions

The full operation vocabulary has ~110 distinct types, but the 15 most frequent account for >90% of all steps. Focus on those.

### Connection to the 5×5 Matrix

Depth is not a replacement for the 5×5 matrix — it is a **within-cell diagnostic**. The two analyses compose naturally:

- The 5×5 matrix tells you **which type combinations** cause the biggest BLIP vs. ViLT gap
- The depth analysis tells you **whether that gap grows with reasoning complexity** within each cell

For example, suppose `query × rel` shows a large BLIP advantage in the matrix. The depth analysis can reveal whether that advantage is:
- **Uniform across depths** → BLIP's language model simply generates better relation-referencing answers
- **Concentrated at depth ≥4** → BLIP handles multi-hop relational queries better specifically, while both models are similar on simple depth-2 rel queries

This kind of finding would be a strong, specific claim for your proposal's contribution statement.

### Expected Findings (Hypotheses)

1. **Both models degrade with depth**, but the rate differs. ViLT (encoder-only, no generative reasoning) is expected to degrade faster at depth ≥5.
2. **Depth-2 `global` questions** (always `select → query scene-level property`) may be the easiest category — both models should perform near ceiling here.
3. **Depth-7 `logical × obj` questions** (nested OR over filtered object sets) are likely the hardest — these require evaluating existence of two separately filtered object sets and combining with Boolean logic.
4. **The `n_relate` feature** (number of relation-traversal steps) may be more predictive of error than raw depth, since relation steps require spatial grounding while other steps (filter, query) are more semantic.

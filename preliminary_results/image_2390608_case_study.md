# Case Study: Image 2390608 — GQA Question Taxonomy Illustration

This image is used to illustrate how GQA's two-axis taxonomy generates a rich, diverse set of questions from a single scene. It is a photograph of a zoo scene: two giraffes (one prominently in frame), a stone wall and fence, and a group of visitors (a lady, a boy, a woman, and another woman) standing on a viewing platform in the background.

**Image dimensions:** 500 × 332 px
**Total questions in val_balanced split:** 30

---

## All 30 Questions — Sorted by Type

| # | Structural | Semantic | Question | Answer |
|---|---|---|---|---|
| 1 | query | attr | Which color do you think the bag on the left of the photo is? | brown |
| 2 | query | attr | What is the color of the sweater the woman is wearing? | blue |
| 3 | query | attr | On which side of the image is the lady? | left |
| 4 | query | attr | What color is the shirt? | pink |
| 5 | query | attr | On which side of the image is the brown bag? | left |
| 6 | query | cat | Which kind of animal is brown? | giraffe |
| 7 | query | rel | Who is wearing the shirt? | woman |
| 8 | query | rel | What is in front of the giraffe? | rock |
| 9 | query | rel | What's in front of the giraffe? | rock |
| 10 | query | rel | What is the woman that is to the left of the bag wearing? | sweater |
| 11 | query | rel | What is the boy wearing? | pants |
| 12 | query | rel | Who do you think is wearing a sweater? | woman |
| 13 | query | rel | Who is the fence in front of? | people |
| 14 | verify | attr | Does the grass look green? | yes |
| 15 | verify | attr | Is the brown animal on the right? | yes |
| 16 | verify | obj | Are there any blue bags? | no |
| 17 | verify | rel | Is there a fence near the people the trees are behind of? | yes |
| 18 | verify | rel | Is the boy wearing jeans? | no |
| 19 | verify | rel | Is there an airplane in front of the bare trees? | no |
| 20 | verify | rel | Is there a skateboard to the left of the person that is wearing trousers? | no |
| 21 | verify | rel | Are there women to the left of the person with the bag? | yes |
| 22 | verify | rel | Are there giraffes near the gray stone? | yes |
| 23 | logical | attr | Is this giraffe tall and white? | no |
| 24 | logical | obj | Are there either chickens or tents? | no |
| 25 | logical | obj | Are there both zebras and giraffes in the photo? | no |
| 26 | choose | rel | Is the boy to the left or to the right of the bag the lady is with? | right |
| 27 | choose | rel | Is the giraffe to the right or to the left of the fence that is in front of the people? | right |
| 28 | choose | rel | Is the lady to the right or to the left of the boy that is to the left of the giraffe? | left |
| 29 | choose | rel | Is the lady to the right or to the left of the people near the fence? | right |
| 30 | compare | attr | Who is older, the lady or the boy? | lady |

---

## Scene Graph: Objects, Bounding Boxes, and Relations

The GQA scene graph for this image encodes **25 objects**, each with:
- A **name** (the object label)
- A **bounding box** in pixel coordinates: `(x, y, w, h)` where `(x, y)` is the top-left corner
- Zero or more **attributes** (color, size, texture, etc.)
- Zero or more **directed relations** to other objects (spatial or functional)

The image is 500 × 332 px. All coordinates are in pixels.

| Object ID | Name | x | y | w | h | Attributes | Key Relations |
|---|---|---:|---:|---:|---:|---|---|
| 1248688 | giraffe | 330 | 29 | 96 | 272 | brown, skinny, tall | in front of trees; near stone; to the right of fence |
| 1248687 | fence | 1 | 114 | 373 | 105 | — | in front of people; near people |
| 1248685 | stone | 115 | 280 | 239 | 51 | gray | near giraffe |
| 3827882 | rock | 312 | 263 | 187 | 71 | — | in front of giraffe |
| 3827899 | rock | 105 | 227 | 109 | 75 | large | to the left of giraffe |
| 3827900 | wall | 0 | 216 | 365 | 116 | — | to the left of giraffe |
| 1248686 | trees | 371 | 0 | 127 | 157 | brown, bare | behind people; behind fence |
| 3827877 | trees | 35 | 0 | 216 | 117 | blurry | to the left of giraffe |
| 3827880 | grass | 357 | 172 | 140 | 114 | green | — |
| 3827905 | woman | 1 | 80 | 59 | 117 | — | wearing sweater |
| 3827887 | lady | 84 | 75 | 50 | 121 | — | with bag; to the left of boy |
| 3827903 | boy | 183 | 107 | 44 | 87 | — | wearing pants |
| 3827901 | woman | 228 | 78 | 53 | 113 | — | wearing shirt |
| 1248689 | people | 9 | 77 | 77 | 121 | — | near fence; in front of trees |
| 3827906 | sweater | 9 | 85 | 70 | 60 | blue | — |
| 3827902 | shirt | 232 | 97 | 49 | 48 | pink | — |
| 3827904 | pants | 186 | 144 | 37 | 48 | colorful | — |
| 3827888 | bag | 116 | 106 | 18 | 27 | brown | with lady |
| 3827886 | hair | 189 | 105 | 20 | 13 | brown | — |
| 3827872 | hair | 240 | 80 | 29 | 32 | black | — |
| 3827891 | horn | 358 | 28 | 20 | 35 | — | — |
| 3827894 | horn | 381 | 29 | 22 | 37 | — | — |
| 3827895 | ear | 328 | 51 | 35 | 31 | — | — |
| 3827898 | ear | 400 | 57 | 27 | 25 | — | — |
| 3827907 | eye | 353 | 82 | 20 | 12 | — | — |

---

## What the Scene Graph Gives Us for Error Analysis

The scene graph is far more than metadata — it provides **pixel-level spatial grounding** for every entity and relationship mentioned in the questions. This makes it a powerful tool for diagnosing *why* a model gets a question wrong.

### 1. Bounding Box Overlap → Visual Ambiguity Score

Each object has a bounding box. When two objects that are both referenced in a question have **heavily overlapping bounding boxes**, the question is visually harder — the model must attend to a cluttered region. We can compute IoU (Intersection over Union) between referenced objects as a proxy for visual difficulty.

**Example from this image:**
Q18: *"Is the boy wearing jeans?"* references `boy` (x=183, y=107, w=44, h=87) and implicitly his clothing. The boy's bounding box is small (44×87 px, ~3% of image area) and located in the middle of the frame, partially overlapping with the lady and the bag. A model that attends to the wrong person (e.g., the woman at x=228) would predict `yes` (she is wearing a pink shirt → no jeans visible) and still get it wrong for the right region.

### 2. Object Size → Resolution Difficulty

Small objects are harder to classify. We can flag questions about small objects (bounding box area < 5% of image) as **resolution-hard** and analyze whether models make more errors on these.

| Object | Box area (px²) | % of image | Likely hard? |
|---|---:|---:|---|
| bag | 18 × 27 = 486 | 0.29% | ⚠️ Yes |
| hair (brown) | 20 × 13 = 260 | 0.16% | ⚠️ Yes |
| hair (black) | 29 × 32 = 928 | 0.56% | ⚠️ Yes |
| eye | 20 × 12 = 240 | 0.14% | ⚠️ Yes |
| giraffe | 96 × 272 = 26,112 | 15.7% | ✅ Easy |
| fence | 373 × 105 = 39,165 | 23.6% | ✅ Easy |

Questions about the `bag` (Q1, Q5, Q26), `hair` (implicitly in Q7), or `eye` are objectively harder due to small object size, and model errors there are more forgivable.

### 3. Relation Chains → Reasoning Depth

Some questions require the model to traverse a **chain of relations** in the scene graph. The length of this chain is a measure of reasoning depth.

**Example — Q28:** *"Is the lady to the right or to the left of the boy that is to the left of the giraffe?"*
This requires: locate `giraffe` → find `boy` that is `to the left of` giraffe → find `lady` relative to that `boy`. That is a **depth-2 relational chain**. Compare to Q3: *"On which side of the image is the lady?"* — depth-0, no relations needed.

We can annotate each question with its **relation chain depth** from the scene graph and correlate it with model accuracy. The hypothesis: accuracy drops as chain depth increases, and the drop is steeper for ViLT (encoder-only, weaker relational reasoning) than for BLIP (generative, better at following multi-step language descriptions).

### 4. Spatial Layout → Attention Map Validation

For models that expose attention weights (ViLT's cross-attention, BLIP's cross-attention layers), we can overlay the ground-truth bounding box of the question's target object onto the image and measure whether the model's peak attention region overlaps with it.

The scene graph gives us the **ground-truth attention target** for free:

| Question | Target object | Ground-truth bbox |
|---|---|---|
| Q4 "What color is the shirt?" | shirt | x=232, y=97, w=49, h=48 |
| Q8 "What is in front of the giraffe?" | rock (3827882) | x=312, y=263, w=187, h=71 |
| Q14 "Does the grass look green?" | grass | x=357, y=172, w=140, h=114 |
| Q1 "What color is the bag?" | bag | x=116, y=106, w=18, h=27 |
| Q6 "Which kind of animal is brown?" | giraffe | x=330, y=29, w=96, h=272 |

This enables a quantitative **attention alignment score**: for a given question, what fraction of the model's top-k attention weight falls inside the ground-truth bounding box? A model that answers correctly but attends to the wrong region may be using a shortcut (e.g., guessing common colors); a model that attends correctly but answers wrong has a reasoning failure, not a grounding failure. This distinction is valuable for error analysis.

### 5. Known Data Quality Caveats in This Image

- **Q30 "Who is older, the lady or the boy?" → `lady`:** Age comparison cannot be reliably inferred from visual features alone. This question is generated from the scene graph label `lady` vs. `boy`, which implies age by naming convention — but it is not visually verifiable in the photograph. Model errors here may not reflect visual understanding failures.
- **Duplicate questions (Q8/Q9):** *"What is in front of the giraffe?"* and *"What's in front of the giraffe?"* are lexically distinct but semantically identical. GQA generates multiple phrasings of the same underlying graph query. Both should have the same model output; discrepancies would indicate sensitivity to surface form.

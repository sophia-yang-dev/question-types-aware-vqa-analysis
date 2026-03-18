# Question-Type-Aware Evaluation of Pretrained VQA Models on GQA

---

## Abstract

Visual Question Answering (VQA) models are typically evaluated using a single aggregate accuracy score over mixed question types, which obscures important variation in model capability across different reasoning demands. In this work, we propose a **question-type-aware evaluation framework** for comparing two pretrained VQA models — BLIP \cite{Li2022BLIP} and ViLT \cite{Kim2021ViLT} — on the GQA dataset \cite{Hudson2019GQA}. Exploiting GQA's native two-axis taxonomy (structural type × semantic type), we evaluate both models across a **5×5 matrix of 15 question categories** that systematically cross reasoning form (query, verify, logical, choose, compare) with visual content target (relational, attribute, object, category, scene-level). Beyond accuracy, we introduce a secondary error analysis based on **functional program depth** — a per-question measure of reasoning chain complexity already annotated in GQA — to diagnose whether model errors are driven by perceptual failure or multi-step reasoning breakdown. Our preliminary analysis reveals that ViLT covers 96.6% of GQA questions with its fixed vocabulary, that `logical` questions are structurally 2× deeper than all other types, and that BLIP outperforms ViLT on a 50-question pilot (56% vs. 48%) with the gap expected to be largest on open-ended relational queries. The results will yield concrete insights into how architectural choices — generative decoding vs. fixed-label classification — interact with question type and reasoning depth.

---

## 1. Introduction

Visual Question Answering is a fundamental vision-language task that requires models to jointly understand natural language questions and visual scenes \cite{Antol2015VQA}. Despite rapid progress driven by large-scale pretraining \cite{Radford2021CLIP, Li2022BLIP, Kim2021ViLT}, standard VQA evaluation conflates highly heterogeneous question types into a single accuracy number. A model that answers *"Is the cat on the left or the right?"* and *"Are there both kites and umbrellas in this scene?"* is measured identically, even though the two questions demand fundamentally different perceptual and reasoning capabilities.

This conflation makes it difficult to answer questions that matter for model understanding: *Does BLIP's generative decoder help on open-ended retrieval but not on yes/no verification? Does ViLT's classification head perform comparably when the answer space is closed? Do both models degrade as questions require longer reasoning chains?*

We address these questions through a systematic, question-type-aware evaluation on GQA \cite{Hudson2019GQA}, a large-scale dataset whose questions are auto-generated from visual scene graphs and come with rich annotations — including a two-axis question taxonomy and per-question functional programs encoding the exact reasoning steps required. This gives us an unusually clean experimental setup: every question has a principled type label and a measurable reasoning complexity score, without any manual annotation effort on our part.

**Our contributions are:**
1. A **5×5 evaluation matrix** covering 15 question type combinations (132,062 questions), yielding fine-grained accuracy profiles for BLIP and ViLT beyond what aggregate VQA benchmarks report.
2. A **capability-based grouping** of the 15 cells into interpretable clusters (open-ended retrieval, binary perception, constrained choice; relational reasoning, attribute recognition, object detection, categorization, scene understanding).
3. A **program-depth error analysis** using GQA's built-in functional program annotations, which allows us to separate perceptual failures from multi-step reasoning failures — a diagnostic not possible on standard VQA benchmarks.

---

## 2. Related Work

### 2.1 Visual Question Answering and Benchmarks

VQA was introduced by Antol et al. \cite{Antol2015VQA}, who proposed joint image-question understanding as a test of AI systems. VQA v2 \cite{Goyal2017VQA2} improved upon the original by balancing yes/no answers to reduce language bias. GQA \cite{Hudson2019GQA} extended this further by grounding every question in a scene graph, enabling compositional and relational questions with guaranteed visual groundedness. GQA's two-axis taxonomy (structural × semantic types) and functional program annotations are unique among VQA datasets and are the foundation of our evaluation design.

### 2.2 Vision-Language Pretrained Models

Early vision-language models such as ViLBERT \cite{Lu2019ViLBERT} used region-based object detectors (e.g., Faster R-CNN) as the visual backbone, adding significant computational overhead. ViLT \cite{Kim2021ViLT} eliminated region supervision entirely, encoding images as flat patch sequences with a single Transformer, drastically reducing inference time while maintaining competitive accuracy. ViLT is trained on VQA v2 and uses a **fixed classification head** over 3,129 answer labels, making it an encoder-only classifier.

BLIP \cite{Li2022BLIP} takes a different approach: bootstrapped pretraining on web-scale image-text data with a **generative language model decoder**, enabling free-form answer generation rather than label classification. BLIP-2 \cite{Li2023BLIP2} further extended this with frozen image encoders and large language models, though we focus on the base BLIP model for tractability.

### 2.3 Type-Disaggregated VQA Analysis

Prior work has noted that VQA accuracy varies substantially across question types. Hudson and Manning \cite{Hudson2019GQA} reported type-level accuracy for their own GQA baseline models. However, most subsequent model papers report only aggregate accuracy. Our work is closest in spirit to analysis papers that break down model performance across question categories \cite{Hudson2019GQA}, but goes further by (1) crossing two taxonomy axes into a 2D matrix, (2) comparing two architecturally distinct pretrained models, and (3) introducing reasoning chain depth as a third explanatory variable. This is not a re-implementation of any single paper — it is a structured comparative analysis using existing pretrained models on an existing dataset, with a novel evaluation framework.

---

## 3. Methods

### 3.1 Models

We compare two pretrained VQA models that represent contrasting architectural philosophies:

**ViLT** (`dandelin/vilt-b32-finetuned-vqa`, \cite{Kim2021ViLT}) encodes image patches and question tokens jointly in a single Transformer encoder, then applies a linear classification head over a fixed vocabulary of $|\mathcal{V}_\text{ViLT}| = 3{,}129$ answer labels. Given image $I$ and question $q$, the predicted answer is:
$$\hat{a}_\text{ViLT} = \arg\max_{a \in \mathcal{V}_\text{ViLT}} \; W \cdot \text{Transformer}([P(I); T(q)])$$
where $P(I)$ are image patch embeddings and $T(q)$ are question token embeddings. ViLT cannot produce answers outside $\mathcal{V}_\text{ViLT}$.

**BLIP** (`Salesforce/blip-vqa-base`, \cite{Li2022BLIP}) uses a ViT image encoder and an autoregressive language model decoder to generate the answer token-by-token. Given image $I$ and question $q$:
$$\hat{a}_\text{BLIP} = \arg\max_{a} \; \prod_{t=1}^{|a|} P(a_t \mid a_{<t}, \text{ViT}(I), T(q))$$
BLIP's answer is unconstrained — it can produce any string, including multi-word answers, paraphrases, and out-of-vocabulary strings.

The key contrast: **ViLT is a closed-vocabulary classifier; BLIP is an open-vocabulary generator.** This architectural difference has direct, testable implications across question types, as discussed in Section 3.3.

### 3.2 The 5×5 Evaluation Matrix

GQA annotates every question with two type labels, giving a natural two-axis evaluation design.

**Structural types** (the reasoning form — what the model must *do*):
- **query** — open-ended retrieval; no constraints on the answer
- **verify** — binary yes/no judgment about a stated proposition
- **logical** — Boolean combination (AND/OR) of two or more sub-propositions
- **choose** — forced binary choice between two options stated in the question
- **compare** — cross-object attribute comparison (same/different)

**Semantic types** (the visual content target — what the model must *see*):
- **rel** — spatial or functional object relationships ("to the left of", "wearing")
- **attr** — object attributes (color, material, size, position)
- **obj** — object existence and identity
- **cat** — object category / semantic class naming
- **global** — holistic scene properties (room type, weather, indoor/outdoor)

Not all 25 combinations exist — GQA's generation grammar produces only **15 populated cells**. The question counts across the full GQA balanced validation set (132,062 questions) are:

|  | **rel** | **attr** | **obj** | **cat** | **global** | **Total** |
|---|---:|---:|---:|---:|---:|---:|
| **query** | 40,278 | 18,092 | — | 7,185 | 2,612 | 68,167 |
| **verify** | 15,602 | 8,053 | 2,879 | — | 879 | 27,413 |
| **logical** | — | 3,411 | 12,685 | — | — | 16,096 |
| **choose** | 5,754 | 8,488 | — | 1,431 | 556 | 16,229 |
| **compare** | — | 4,157 | — | — | — | 4,157 |
| **Total** | 61,634 | 42,201 | 15,564 | 8,616 | 4,047 | **132,062** |

The dataset is dominated by `query×rel` (30.5%) and `query×attr` (13.7%), with the smallest populated cell being `choose×global` (556 questions, 0.4%).

### 3.3 Capability-Based Groupings

For the secondary narrative analysis, we group the 15 cells along each axis into interpretable capability clusters:

**By structural axis (what kind of reasoning):**
- **S1 — Open-ended retrieval** (`query` row, $n=68{,}167$): model must generate an answer from scratch. Architecturally favors BLIP.
- **S2 — Binary perception** (`verify` + `logical` rows, $n=43{,}509$): all answers are yes/no. Level playing field for both models; `logical` additionally tests Boolean composition.
- **S3 — Constrained choice** (`choose` + `compare` rows, $n=20{,}386$): answer space is revealed in the question. May favor ViLT's classification paradigm.

**By semantic axis (what visual skill is needed):**
- **V1 — Relational reasoning** (`rel` column, $n=61{,}634$): spatial/functional language grounding
- **V2 — Attribute recognition** (`attr` column, $n=42{,}201$): fine-grained visual feature extraction
- **V3 — Object detection** (`obj` column, $n=15{,}564$): object existence verification
- **V4 — Categorization** (`cat` column, $n=8{,}616$): semantic category naming
- **V5 — Scene understanding** (`global` column, $n=4{,}047$): holistic scene-level comprehension

### 3.4 Reasoning Chain Depth Analysis

Every GQA question is annotated with a **functional program** — a DAG of named reasoning operations used to generate the question from the scene graph. This is stored directly in `val_balanced_questions.json` and requires no inference. We define **program depth** as the number of operations in the program:

$$d(q) = |\texttt{q["semantic"]}|$$

Each operation has a name (e.g., `select`, `relate`, `query`, `exist`, `filter color`, `and`, `or`) and explicit dependencies on prior steps. Three example programs from a single image illustrate the range:

```
depth 2 — "What color is the shirt?"
  [0] select: shirt → [1] query: color

depth 3 — "Who is the fence in front of?"
  [0] select: fence → [1] relate: person, in front of → [2] query: name

depth 5 — "Are there both zebras and giraffes in the photo?"
  [0] select: giraffe → [1] exist
  [2] select: zebra  → [3] exist → [4] and([1],[3])
```

Depth is not uniformly distributed. In our dataset: 27.8% of questions are depth 2, 47.6% are depth 3, and `logical` questions have mean depth 5.32 versus ≤3.0 for all other structural types — because logical questions require running two independent sub-programs and combining them with AND/OR, effectively doubling program length.

We propose computing additional features from the program beyond raw depth:

```python
def program_features(q):
    ops = [step['operation'] for step in q['semantic']]
    return {
        'depth':        len(ops),
        'n_relate':     ops.count('relate'),
        'n_filter':     sum(1 for op in ops if op.startswith('filter')),
        'has_logical':  any(op in ('and', 'or') for op in ops),
        'n_selects':    ops.count('select'),
    }
```

`n_relate` (number of relation-traversal steps) is expected to be a stronger predictor of error than raw depth alone, since spatial relation grounding is a harder perceptual operation than attribute lookup.

---

## 4. Experimental Setup

### 4.1 Dataset

We use the **GQA balanced validation split** (132,062 questions, 1,469 unique answer strings). The balanced split ensures that yes/no questions are 50/50 distributed, preventing majority-class trivial baselines. We do not use the unbalanced split or test split.

**Preliminary findings from dataset analysis:**
- ViLT's fixed vocabulary covers **96.6%** of all questions at the question level (≥93% in every cell), confirming that ViLT's closed vocabulary is not a structural barrier for any proposed category.
- The GQA balanced split contains **no "how many" questions** and **no questions with digit-only answers**, confirming that counting ability cannot be evaluated on this split.
- Scene graphs provide bounding box annotations for all 25 objects in the case-study image (and similarly for all images), enabling attention map alignment analysis in the error analysis phase.

### 4.2 Implementation

Both models are loaded from HuggingFace `transformers`:
- `Salesforce/blip-vqa-base` (BLIP)
- `dandelin/vilt-b32-finetuned-vqa` (ViLT)

Both models are used in inference-only mode (no fine-tuning on GQA). We run in `eval()` mode with `torch.no_grad()`. From our preliminary benchmarking on 50 questions (CPU):

| Model | Mean inference time | Est. full-set runtime |
|---|---|---|
| BLIP | 0.266 s/sample | ~9.8 hours (CPU) |
| ViLT | 0.063 s/sample | ~2.3 hours (CPU) |

ViLT is 4.2× faster due to classification vs. autoregressive generation. To make full evaluation tractable, we will use a **stratified random sample of 10,000 questions**, sampling proportionally from each of the 15 cells (minimum ~76 questions per cell for the smallest cell, `choose×global` with 556 total). This reduces BLIP runtime to under 45 minutes while preserving the distributional structure of the full dataset.

### 4.3 Evaluation Metric

We use **exact-match accuracy** after the following normalization:

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
```

This handles case normalization, number-word/digit equivalence, and leading article removal. Our 50-question pilot found no case-only mismatches and GQA answers are already single clean tokens, so normalization has minimal effect on this dataset — but it prevents spurious mismatches when BLIP generates slightly different surface forms.

### 4.4 Comparisons and Analysis Plan

**Primary comparison:** BLIP vs. ViLT accuracy in each of the 15 populated cells of the 5×5 matrix. Reported as:
1. Two 5×5 accuracy heatmaps (one per model)
2. One 5×5 gap heatmap (BLIP $-$ ViLT), signed so positive = BLIP wins
3. Row marginals (accuracy averaged over each structural type) and column marginals (averaged over each semantic type)

**Secondary comparison:** Accuracy aggregated per capability group (S1/S2/S3 and V1/V2/V3/V4/V5) with 95% confidence intervals from bootstrap resampling.

**Tertiary analysis — program depth:** For each model, plot exact-match accuracy as a function of depth bin $\{2, 3, 4, 5, \geq 6\}$, both overall and within each structural type. This answers whether model degradation with question complexity is uniform or concentrated in particular cell regions of the matrix.

**Preliminary pilot results (50 questions, mixed types):**
- BLIP: 56.0% exact-match accuracy
- ViLT: 48.0% exact-match accuracy
- All 26 ViLT errors were in-vocabulary failures (correct answer existed in ViLT's label set but model predicted wrong), confirming that vocabulary coverage is not the bottleneck

---

## 5. Timeline and Work Breakdown

| Week | Milestone | Owner |
|---|---|---|
| Week 1–2 | Full inference run: stratified 10K sample, both models, record predictions + timing | All |
| Week 2 | Compute 5×5 accuracy matrix; generate heatmaps | Member A |
| Week 3 | Program depth feature extraction; accuracy-vs-depth plots | Member B |
| Week 3 | Capability group analysis (S1–S3, V1–V5); confidence intervals | Member A |
| Week 4 | Error case study: bounding-box / attention-alignment analysis on scene graph | Member B |
| Week 4–5 | Write-up: results, discussion, conclusion | All |
| **Milestone checkpoint** | 5×5 matrix complete + depth analysis plots ready | End of Week 3 |

**By the milestone**, we plan to have: (1) all inference results collected, (2) the full 5×5 matrix with heatmaps, (3) the accuracy-vs-depth plot for both models, and (4) the capability group summary. The error case study and final write-up will follow in the remaining weeks.

---

## References

```bibtex
@inproceedings{Antol2015VQA,
  title={VQA: Visual Question Answering},
  author={Antol, Stanislaw and Agrawal, Aishwarya and Lu, Jiasen and Mitchell, Margaret
          and Batra, Dhruv and Zitnick, C. Lawrence and Parikh, Devi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2015},
  pages={2425--2433}
}

@inproceedings{Goyal2017VQA2,
  title={Making the {V} in {VQA} Matter: Elevating the Role of Image Understanding
         in Visual Question Answering},
  author={Goyal, Yash and Khot, Tejas and Summers-Stay, Douglas and Batra, Dhruv
          and Parikh, Devi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern
             Recognition (CVPR)},
  year={2017},
  pages={6904--6913}
}

@inproceedings{Hudson2019GQA,
  title={{GQA}: A New Dataset for Real-World Visual Reasoning and Compositional
         Question Answering},
  author={Hudson, Drew A. and Manning, Christopher D.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
             Recognition (CVPR)},
  year={2019},
  pages={6700--6711}
}

@inproceedings{Lu2019ViLBERT,
  title={{ViLBERT}: Pretraining Task-Agnostic Visiolinguistic Representations for
         Vision-and-Language Tasks},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2019},
  pages={13--23}
}

@inproceedings{Kim2021ViLT,
  title={{ViLT}: Vision-and-Language Transformer Without Convolution or Region Supervision},
  author={Kim, Wonjae and Son, Bokyung and Kim, Ildoo},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
  year={2021},
  pages={5583--5594}
}

@inproceedings{Radford2021CLIP,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya
          and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda
          and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
  year={2021},
  pages={8748--8763}
}

@inproceedings{Li2022BLIP,
  title={{BLIP}: Bootstrapping Language-Image Pre-training for Unified Vision-Language
         Understanding and Generation},
  author={Li, Junnan and Li, Dongxu and Savarese, Silvio and Hoi, Steven},
  booktitle={Proceedings of the 39th International Conference on Machine Learning (ICML)},
  year={2022},
  pages={12888--12900}
}

@inproceedings{Li2023BLIP2,
  title={{BLIP-2}: Bootstrapping Language-Image Pre-training with Frozen Image Encoders
         and Large Language Models},
  author={Li, Junnan and Li, Dongxu and Savarese, Silvio and Hoi, Steven C. H.},
  booktitle={Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year={2023},
  pages={19730--19742}
}
```

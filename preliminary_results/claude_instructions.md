# GQA Preliminary Analysis — Claude Code Instructions

## Context

I am working on an NLP course project comparing two pretrained VQA models — **BLIP** (`Salesforce/blip-vqa-base`) and **ViLT** (`dandelin/vilt-b32-finetuned-vqa`) — on the GQA dataset. The goal of this session is to run preliminary analyses to inform my project proposal. Please complete the analyses below **in order**, and ask me for file paths or clarification before running anything that touches my local filesystem.

---

## Setup

Before starting, please:

1. Ask me for the path to my GQA balanced validation questions JSON file (`val_balanced_questions.json`)
2. Ask me for the path to the folder where my GQA images are stored (the .zip may still be extracting — help me extract a **sample of 100 random images** from it if needed, rather than waiting for the full extraction)
3. Confirm the following Python packages are available, and install any that are missing:

```
transformers torch torchvision pillow matplotlib pandas numpy tqdm
```

4. Create an output folder called `preliminary_results/` in the current working directory to save all figures and tables

---

## Analysis 1: Extract a Sample of Images from the ZIP

> Run this first if the full ZIP has not finished extracting yet, so we can proceed with timing analysis in parallel.

- Open the GQA images ZIP file (I will give you the path)
- List the contents to understand the internal folder structure
- Randomly sample **100 image filenames** from the ZIP
- Extract only those 100 images into `preliminary_results/sample_images/`
- Print the total number of images found in the ZIP and confirm the 100 were extracted successfully
- Save the list of sampled image IDs (filenames without extension) to `preliminary_results/sampled_image_ids.txt`

---

## Analysis 2: Category Distribution in GQA Balanced Validation Split

Load `val_balanced_questions.json`. Each entry is keyed by question ID and contains at minimum:

```
question, answer, types.structural, types.semantic
```

The structural types are: `verify`, `query`, `choose`, `logical`, `compare`  
The semantic types are: `obj`, `attr`, `rel`, `cat`, `global`

Please produce the following:

**2a. Structural type distribution**
- Count and percentage of each structural type across all questions
- Save as a bar chart: `preliminary_results/structural_type_distribution.png`

**2b. Semantic type distribution**
- Count and percentage of each semantic type across all questions
- Save as a bar chart: `preliminary_results/semantic_type_distribution.png`

**2c. Query breakdown by semantic type**
- For questions where `structural == "query"` only, show the breakdown by semantic type
- This is important because "query" is the largest bucket and we plan to split it further

**2d. Counting questions**
- Identify questions where `question.lower().startswith("how many")`
- Report: how many there are, what percentage of total, and what structural/semantic types GQA assigns to them (they may span multiple GQA types)
- This tells us whether our custom "counting" category overlaps cleanly with existing GQA types

**2e. Example questions per structural type**
- Print 3 randomly sampled questions with their ground-truth answers for each of the 5 structural types
- This gives intuitive feel for the data

**2f. Summary table**
- Print a clean summary table of all category counts and percentages to the terminal

---

## Analysis 3: ViLT Answer Vocabulary Coverage on GQA

This analysis determines how many GQA questions can in principle be answered correctly by ViLT, given that ViLT is a classification model with a fixed vocabulary from VQA v2.0 training.

**3a. Load ViLT vocabulary**

```python
from transformers import ViltForQuestionAnswering
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_labels = set(model.config.id2label.values())
print(f"ViLT vocabulary size: {len(vilt_labels)}")
```

**3b. Load GQA answers**
- Collect all ground-truth answers from the validation JSON
- Compute: number of unique answer strings in GQA

**3c. Coverage metrics — compute all of the following:**

| Metric | Description |
|---|---|
| Unique answer coverage | % of unique GQA answer strings that appear in ViLT vocab |
| Question-level coverage | % of questions whose ground-truth answer is in ViLT vocab (weighted by frequency) |
| Coverage by structural type | Question-level coverage for each of the 5 structural types |
| Coverage by semantic type | Question-level coverage for each of the 5 semantic types |
| Coverage for counting questions | Question-level coverage for "how many" questions specifically |

**3d. Top uncovered answers**
- List the **top 30 most frequent GQA answers** that do NOT appear in ViLT's vocabulary
- This tells us what kinds of answers ViLT structurally cannot produce

**3e. Save outputs**
- Save the full coverage table to `preliminary_results/vilt_coverage.csv`
- Save a bar chart of question-level coverage by structural type: `preliminary_results/vilt_coverage_by_structural_type.png`
- Save a bar chart of question-level coverage by semantic type: `preliminary_results/vilt_coverage_by_semantic_type.png`

---

## Analysis 4: Inference Time Benchmarking

> Requires the 100 sampled images from Analysis 1. If those are not ready yet, skip to Analysis 5 and come back.

Using the sampled images, randomly select **50 questions from the validation JSON whose image IDs match the sampled images**. Run both models on these 50 questions and measure inference time.

**4a. Setup both models**

```python
# BLIP
from transformers import BlipProcessor, BlipForQuestionAnswering
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# ViLT
from transformers import ViltProcessor, ViltForQuestionAnswering
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
```

- Run on CPU (note whether GPU is available)
- Set both models to eval mode (`model.eval()`)
- Use `torch.no_grad()` during inference

**4b. Timing measurement**
- For each of the 50 questions, time the full inference pipeline: image loading → preprocessing → model forward pass → answer extraction
- Use `time.perf_counter()` for timing, not `time.time()`
- Record per-sample times for both models

**4c. Report the following for each model:**
- Mean inference time per sample (seconds)
- Median inference time per sample
- Std deviation
- Min and max
- Estimated time to run the full balanced validation split (~132K questions) extrapolated from the mean

**4d. Practical issues log**
- Note any errors, warnings, or unexpected behaviors encountered
- Note image size or format requirements for each model
- Note any differences in how images need to be preprocessed

Save timing results to `preliminary_results/inference_timing.csv`

---

## Analysis 5: Answer Format and Normalization Analysis

Using the same 50 questions from Analysis 4 (or a different 50 if images aren't ready), run both models and compare outputs side by side against GQA ground truth.

**5a. Side-by-side comparison table**

For each of the 50 questions, collect:
- The question text
- The ground truth GQA answer
- BLIP's raw predicted answer
- ViLT's raw predicted answer (top label)
- Whether BLIP's answer exactly matches ground truth (case-insensitive)
- Whether ViLT's answer exactly matches ground truth (case-insensitive)

Print the first 20 rows as a readable table and save all 50 to `preliminary_results/answer_comparison.csv`

**5b. Answer format observations — check and report on:**

| Issue | What to check |
|---|---|
| Case differences | Does "Yes" vs "yes" cause mismatches? |
| Number format | Does "2" vs "two" cause mismatches? |
| Article differences | Does "a dog" vs "dog" cause mismatches? |
| Multi-word answers | How often does BLIP produce multi-word answers? Does GQA have multi-word ground truth? |
| ViLT vocabulary misses | For questions where ViLT is wrong, is the correct answer in ViLT's vocab at all? |

**5c. Recommend a normalization strategy**
Based on what you observe, suggest a simple answer normalization function I should apply to both models' outputs and the ground truth before computing accuracy. Implement it and show how it affects exact-match accuracy on the 50-sample test.

---

## Analysis 6: Category Balance Sanity Check

Based on the full validation JSON (no images needed), check the size of each of our 9 proposed evaluation categories:

| # | Category | How to identify |
|---|---|---|
| 1 | Verify (yes/no) | `structural == "verify"` |
| 2 | Choose | `structural == "choose"` |
| 3 | Logical | `structural == "logical"` |
| 4 | Compare | `structural == "compare"` |
| 5 | Object query | `structural == "query"` AND `semantic == "obj"` |
| 6 | Attribute query | `structural == "query"` AND `semantic == "attr"` |
| 7 | Relation query | `structural == "query"` AND `semantic == "rel"` |
| 8 | Category query | `structural == "query"` AND `semantic == "cat"` |
| 9 | Counting | `question.lower().startswith("how many")` |

Note: Category 9 (counting) may overlap with categories 5–8. Report the overlap explicitly.

**For each category, report:**
- Total question count
- Percentage of full validation set
- Flag with ⚠️ if fewer than 500 questions (too small for reliable evaluation)
- Flag with ✅ if more than 5000 questions (well-represented)

Save output to `preliminary_results/category_balance.csv` and print a formatted table.

---

## Final Summary

After completing all analyses, please produce a single file `preliminary_results/SUMMARY.md` containing:

1. **Dataset overview**: total questions in balanced validation split, answer vocabulary size, structural and semantic type breakdown
2. **Category sizes**: the 9-category table from Analysis 6
3. **ViLT coverage**: overall question-level coverage, and coverage by structural type — flag any category below 70% coverage as a concern
4. **Inference timing**: mean time per sample for each model, estimated total runtime for full evaluation
5. **Answer normalization**: the recommended normalization strategy and its effect on accuracy
6. **Key decisions for proposal**: based on all findings, list 3–5 concrete decisions I should make when designing my experiments (e.g., whether to filter by ViLT vocabulary, which categories to merge if too small, whether to use a subset of the validation split, etc.)

This summary will be used directly to inform the experimental design section of my project proposal.
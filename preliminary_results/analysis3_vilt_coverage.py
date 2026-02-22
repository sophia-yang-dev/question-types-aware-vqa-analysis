import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = "/Users/xuefeiyang/Documents/NLP/question-types-aware-vqa-analysis/data/questions1.2/val_balanced_questions.json"
OUT_DIR = "/Users/xuefeiyang/Documents/NLP/question-types-aware-vqa-analysis/preliminary_results"

# 3a. Load ViLT vocabulary
print("Loading ViLT model to extract vocabulary...")
from transformers import ViltForQuestionAnswering
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_labels = set(model.config.id2label.values())
print(f"ViLT vocabulary size: {len(vilt_labels)}")
del model  # free memory

# 3b. Load GQA answers
print("\nLoading GQA val_balanced_questions.json ...")
with open(DATA_PATH) as f:
    data = json.load(f)
questions = list(data.values())
total = len(questions)

gqa_answers = [q['answer'] for q in questions]
unique_gqa_answers = set(gqa_answers)
print(f"Total questions: {total:,}")
print(f"Unique GQA answers: {len(unique_gqa_answers):,}")

# 3c. Coverage metrics
covered_unique = unique_gqa_answers & vilt_labels
unique_coverage = len(covered_unique) / len(unique_gqa_answers)
print(f"\n=== 3c. Coverage Metrics ===")
print(f"Unique answer coverage: {len(covered_unique)}/{len(unique_gqa_answers)} = {100*unique_coverage:.1f}%")

# Question-level coverage (weighted by frequency)
q_covered = sum(1 for a in gqa_answers if a in vilt_labels)
q_coverage = q_covered / total
print(f"Question-level coverage: {q_covered:,}/{total:,} = {100*q_coverage:.1f}%")

# Coverage by structural type
struct_types = ['verify', 'query', 'choose', 'logical', 'compare']
struct_coverage = {}
for t in struct_types:
    pool = [q for q in questions if q['types']['structural'] == t]
    covered = sum(1 for q in pool if q['answer'] in vilt_labels)
    struct_coverage[t] = {'total': len(pool), 'covered': covered, 'pct': 100*covered/len(pool)}

print("\nCoverage by structural type:")
for t, v in struct_coverage.items():
    print(f"  {t:10s}: {v['covered']:6,}/{v['total']:6,} = {v['pct']:.1f}%")

# Coverage by semantic type
sem_types = ['rel', 'attr', 'obj', 'cat', 'global']
sem_coverage = {}
for t in sem_types:
    pool = [q for q in questions if q['types']['semantic'] == t]
    covered = sum(1 for q in pool if q['answer'] in vilt_labels)
    sem_coverage[t] = {'total': len(pool), 'covered': covered, 'pct': 100*covered/len(pool)}

print("\nCoverage by semantic type:")
for t, v in sem_coverage.items():
    print(f"  {t:10s}: {v['covered']:6,}/{v['total']:6,} = {v['pct']:.1f}%")

# Coverage for counting questions
counting_qs = [q for q in questions if q['question'].lower().startswith('how many')]
if counting_qs:
    cov_count = sum(1 for q in counting_qs if q['answer'] in vilt_labels)
    print(f"\nCounting question coverage: {cov_count}/{len(counting_qs)} = {100*cov_count/len(counting_qs):.1f}%")
else:
    print("\nNo 'how many' questions found in balanced val set.")

# 3d. Top uncovered answers
from collections import Counter
answer_freq = Counter(gqa_answers)
uncovered = {a: c for a, c in answer_freq.items() if a not in vilt_labels}
top30_uncovered = sorted(uncovered.items(), key=lambda x: -x[1])[:30]
print("\n=== 3d. Top 30 Most Frequent GQA Answers NOT in ViLT Vocab ===")
for rank, (ans, cnt) in enumerate(top30_uncovered, 1):
    print(f"  {rank:2d}. '{ans}': {cnt:,}")

# 3e. Save outputs
rows = []
for t, v in struct_coverage.items():
    rows.append({'type': 'structural', 'category': t, 'total': v['total'],
                 'covered': v['covered'], 'coverage_pct': round(v['pct'], 1)})
for t, v in sem_coverage.items():
    rows.append({'type': 'semantic', 'category': t, 'total': v['total'],
                 'covered': v['covered'], 'coverage_pct': round(v['pct'], 1)})
rows.append({'type': 'overall', 'category': 'all', 'total': total,
             'covered': q_covered, 'coverage_pct': round(100*q_coverage, 1)})

df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/vilt_coverage.csv", index=False)
print(f"\nSaved vilt_coverage.csv")

# Bar chart by structural type
fig, ax = plt.subplots(figsize=(8, 5))
st_labels = list(struct_coverage.keys())
st_pcts = [struct_coverage[t]['pct'] for t in st_labels]
bars = ax.bar(st_labels, st_pcts, color='mediumseagreen', edgecolor='white')
ax.axhline(70, color='red', linestyle='--', label='70% threshold')
ax.set_ylim(0, 105)
ax.set_title('ViLT Answer Vocab Coverage by Structural Type', fontsize=13)
ax.set_xlabel('Structural Type')
ax.set_ylabel('Coverage (%)')
ax.legend()
for bar, pct in zip(bars, st_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/vilt_coverage_by_structural_type.png", dpi=150)
plt.close()
print("Saved vilt_coverage_by_structural_type.png")

# Bar chart by semantic type
fig, ax = plt.subplots(figsize=(8, 5))
se_labels = list(sem_coverage.keys())
se_pcts = [sem_coverage[t]['pct'] for t in se_labels]
bars = ax.bar(se_labels, se_pcts, color='mediumpurple', edgecolor='white')
ax.axhline(70, color='red', linestyle='--', label='70% threshold')
ax.set_ylim(0, 105)
ax.set_title('ViLT Answer Vocab Coverage by Semantic Type', fontsize=13)
ax.set_xlabel('Semantic Type')
ax.set_ylabel('Coverage (%)')
ax.legend()
for bar, pct in zip(bars, se_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/vilt_coverage_by_semantic_type.png", dpi=150)
plt.close()
print("Saved vilt_coverage_by_semantic_type.png")

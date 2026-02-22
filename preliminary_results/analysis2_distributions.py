import json
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = "/Users/xuefeiyang/Documents/NLP/question-types-aware-vqa-analysis/data/questions1.2/val_balanced_questions.json"
OUT_DIR = "/Users/xuefeiyang/Documents/NLP/question-types-aware-vqa-analysis/preliminary_results"

print("Loading val_balanced_questions.json ...")
with open(DATA_PATH) as f:
    data = json.load(f)
print(f"Total questions: {len(data)}")

questions = list(data.values())

# --- 2a. Structural type distribution ---
struct_counts = {}
for q in questions:
    t = q['types']['structural']
    struct_counts[t] = struct_counts.get(t, 0) + 1

total = len(questions)
print("\n=== 2a. Structural Type Distribution ===")
for t, c in sorted(struct_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:12s}: {c:7,d}  ({100*c/total:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 5))
labels = sorted(struct_counts, key=lambda x: -struct_counts[x])
vals = [struct_counts[l] for l in labels]
pcts = [100*v/total for v in vals]
bars = ax.bar(labels, vals, color='steelblue', edgecolor='white')
ax.set_title('GQA Val Balanced — Structural Type Distribution', fontsize=13)
ax.set_xlabel('Structural Type')
ax.set_ylabel('Question Count')
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/structural_type_distribution.png", dpi=150)
plt.close()
print(f"Saved structural_type_distribution.png")

# --- 2b. Semantic type distribution ---
sem_counts = {}
for q in questions:
    t = q['types']['semantic']
    sem_counts[t] = sem_counts.get(t, 0) + 1

print("\n=== 2b. Semantic Type Distribution ===")
for t, c in sorted(sem_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:10s}: {c:7,d}  ({100*c/total:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 5))
labels_s = sorted(sem_counts, key=lambda x: -sem_counts[x])
vals_s = [sem_counts[l] for l in labels_s]
pcts_s = [100*v/total for v in vals_s]
bars_s = ax.bar(labels_s, vals_s, color='darkorange', edgecolor='white')
ax.set_title('GQA Val Balanced — Semantic Type Distribution', fontsize=13)
ax.set_xlabel('Semantic Type')
ax.set_ylabel('Question Count')
for bar, pct in zip(bars_s, pcts_s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/semantic_type_distribution.png", dpi=150)
plt.close()
print(f"Saved semantic_type_distribution.png")

# --- 2c. Query breakdown by semantic type ---
query_qs = [q for q in questions if q['types']['structural'] == 'query']
query_sem = {}
for q in query_qs:
    t = q['types']['semantic']
    query_sem[t] = query_sem.get(t, 0) + 1

print(f"\n=== 2c. Query Questions ({len(query_qs):,}) — Breakdown by Semantic Type ===")
for t, c in sorted(query_sem.items(), key=lambda x: -x[1]):
    print(f"  {t:10s}: {c:7,d}  ({100*c/len(query_qs):.1f}% of queries)")

# --- 2d. Counting questions ---
counting_qs = [q for q in questions if q['question'].lower().startswith('how many')]
print(f"\n=== 2d. Counting Questions ('how many...') ===")
print(f"  Count: {len(counting_qs):,}  ({100*len(counting_qs)/total:.2f}% of total)")

count_struct = {}
count_sem = {}
for q in counting_qs:
    s = q['types']['structural']
    se = q['types']['semantic']
    count_struct[s] = count_struct.get(s, 0) + 1
    count_sem[se] = count_sem.get(se, 0) + 1

print("  Structural types among counting questions:")
for t, c in sorted(count_struct.items(), key=lambda x: -x[1]):
    print(f"    {t:12s}: {c:5,d}  ({100*c/len(counting_qs):.1f}%)")
print("  Semantic types among counting questions:")
for t, c in sorted(count_sem.items(), key=lambda x: -x[1]):
    print(f"    {t:10s}: {c:5,d}  ({100*c/len(counting_qs):.1f}%)")

# --- 2e. 3 example questions per structural type ---
print("\n=== 2e. Example Questions per Structural Type ===")
random.seed(42)
for stype in ['verify', 'query', 'choose', 'logical', 'compare']:
    pool = [q for q in questions if q['types']['structural'] == stype]
    samples = random.sample(pool, min(3, len(pool)))
    print(f"\n  [{stype.upper()}]")
    for s in samples:
        print(f"    Q: {s['question']}")
        print(f"    A: {s['answer']}")

# --- 2f. Summary table ---
print("\n=== 2f. Summary Table ===")
print(f"{'Type':<15} {'Category':<12} {'Count':>8} {'Percent':>8}")
print("-" * 48)
for t, c in sorted(struct_counts.items(), key=lambda x: -x[1]):
    print(f"{'structural':<15} {t:<12} {c:>8,} {100*c/total:>7.1f}%")
for t, c in sorted(sem_counts.items(), key=lambda x: -x[1]):
    print(f"{'semantic':<15} {t:<12} {c:>8,} {100*c/total:>7.1f}%")
print("-" * 48)
print(f"{'TOTAL':<28} {total:>8,} {'100.0%':>8}")

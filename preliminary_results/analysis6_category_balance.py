import json
import pandas as pd

DATA_PATH = "/Users/xuefeiyang/Documents/NLP/question-types-aware-vqa-analysis/data/questions1.2/val_balanced_questions.json"
OUT_DIR = "/Users/xuefeiyang/Documents/NLP/question-types-aware-vqa-analysis/preliminary_results"

print("Loading val_balanced_questions.json ...")
with open(DATA_PATH) as f:
    data = json.load(f)
questions = list(data.values())
total = len(questions)
print(f"Total questions: {total:,}")

categories = [
    (1, "Verify (yes/no)",   lambda q: q['types']['structural'] == 'verify'),
    (2, "Choose",            lambda q: q['types']['structural'] == 'choose'),
    (3, "Logical",           lambda q: q['types']['structural'] == 'logical'),
    (4, "Compare",           lambda q: q['types']['structural'] == 'compare'),
    (5, "Object query",      lambda q: q['types']['structural'] == 'query' and q['types']['semantic'] == 'obj'),
    (6, "Attribute query",   lambda q: q['types']['structural'] == 'query' and q['types']['semantic'] == 'attr'),
    (7, "Relation query",    lambda q: q['types']['structural'] == 'query' and q['types']['semantic'] == 'rel'),
    (8, "Category query",    lambda q: q['types']['structural'] == 'query' and q['types']['semantic'] == 'cat'),
    (9, "Counting (how many)", lambda q: q['question'].lower().startswith('how many')),
]

rows = []
cat9_ids = set()
for num, name, fn in categories:
    matched = [q for q in questions if fn(q)]
    count = len(matched)
    pct = 100 * count / total
    flag = ''
    if count < 500:
        flag = '⚠️ TOO SMALL'
    elif count > 5000:
        flag = '✅ WELL-REPRESENTED'
    rows.append({'#': num, 'Category': name, 'Count': count, 'Pct': round(pct, 2), 'Flag': flag})
    if num == 9:
        cat9_ids = set(id(q) for q in matched)

# Overlap: counting vs categories 1-8
print("\n=== Analysis 6: Category Balance ===")
print(f"\n{'#':<3} {'Category':<22} {'Count':>8} {'Pct':>7}  Flag")
print("-" * 62)
for r in rows:
    print(f"{r['#']:<3} {r['Category']:<22} {r['Count']:>8,} {r['Pct']:>6.2f}%  {r['Flag']}")
print("-" * 62)
print(f"{'TOTAL':<26} {total:>8,} {'100.00%':>7}")

# Counting overlap with cats 5-8
counting_qs = [q for q in questions if q['question'].lower().startswith('how many')]
print(f"\n=== Counting Question Overlap with Categories 5-8 ===")
print(f"Total counting ('how many') questions: {len(counting_qs)}")
if counting_qs:
    for num, name, fn in categories[4:8]:
        overlap = [q for q in counting_qs if fn(q)]
        print(f"  Overlap with Cat {num} ({name}): {len(overlap)}")
else:
    print("  None found in the balanced val set — 'how many' category yields 0 questions here.")
    print("  NOTE: GQA balanced splits may undersample counting questions by design.")

# Save CSV
df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/category_balance.csv", index=False)
print(f"\nSaved category_balance.csv")

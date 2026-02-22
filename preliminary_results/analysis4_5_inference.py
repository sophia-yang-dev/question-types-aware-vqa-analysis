import json
import os
import random
import time
import csv
import re
import pandas as pd
import torch
from PIL import Image
from transformers import (
    BlipProcessor, BlipForQuestionAnswering,
    ViltProcessor, ViltForQuestionAnswering,
)

DATA_PATH = "data/questions1.2/val_balanced_questions.json"
IMAGES_DIR = "preliminary_results/sample_images"
OUT_DIR = "preliminary_results"
IDS_FILE = "preliminary_results/sampled_image_ids.txt"

random.seed(42)

# Load sampled image IDs
with open(IDS_FILE) as f:
    sampled_ids = set(l.strip() for l in f if l.strip())
print(f"Sampled image IDs loaded: {len(sampled_ids)}")

# Load questions matching sampled images
print("Loading questions...")
with open(DATA_PATH) as f:
    data = json.load(f)

matching_qs = [(qid, q) for qid, q in data.items() if q['imageId'] in sampled_ids]
print(f"Questions matching sampled images: {len(matching_qs)}")

if len(matching_qs) < 50:
    selected = matching_qs
    print(f"WARNING: Only {len(matching_qs)} matching questions, using all of them.")
else:
    selected = random.sample(matching_qs, 50)
print(f"Using {len(selected)} questions for benchmarking")

# GPU check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load models
print("\nLoading BLIP...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device).eval()

print("Loading ViLT...")
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device).eval()

# --- Analysis 4 & 5: Run inference ---
blip_times = []
vilt_times = []
results = []
practical_issues = []

print(f"\nRunning inference on {len(selected)} questions...")
for i, (qid, q) in enumerate(selected):
    img_path = os.path.join(IMAGES_DIR, f"{q['imageId']}.jpg")
    question_text = q['question']
    gt_answer = q['answer']

    if not os.path.exists(img_path):
        practical_issues.append(f"Missing image: {img_path}")
        continue

    # BLIP inference
    try:
        t0 = time.perf_counter()
        image = Image.open(img_path).convert('RGB')
        inputs = blip_processor(image, question_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(**inputs)
        blip_answer = blip_processor.decode(out[0], skip_special_tokens=True)
        blip_time = time.perf_counter() - t0
        blip_times.append(blip_time)
    except Exception as e:
        blip_answer = "ERROR"
        blip_time = None
        practical_issues.append(f"BLIP error on {qid}: {e}")

    # ViLT inference
    try:
        t0 = time.perf_counter()
        image = Image.open(img_path).convert('RGB')
        inputs = vilt_processor(image, question_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = vilt_model(**inputs)
        logits = out.logits
        pred_id = logits.argmax(-1).item()
        vilt_answer = vilt_model.config.id2label[pred_id]
        vilt_time = time.perf_counter() - t0
        vilt_times.append(vilt_time)
    except Exception as e:
        vilt_answer = "ERROR"
        vilt_time = None
        practical_issues.append(f"ViLT error on {qid}: {e}")

    blip_match = (blip_answer.strip().lower() == gt_answer.strip().lower())
    vilt_match = (vilt_answer.strip().lower() == gt_answer.strip().lower())

    results.append({
        'qid': qid,
        'imageId': q['imageId'],
        'question': question_text,
        'gt_answer': gt_answer,
        'blip_answer': blip_answer,
        'vilt_answer': vilt_answer,
        'blip_exact_match': blip_match,
        'vilt_exact_match': vilt_match,
        'blip_time': round(blip_time, 4) if blip_time else None,
        'vilt_time': round(vilt_time, 4) if vilt_time else None,
        'structural': q['types']['structural'],
        'semantic': q['types']['semantic'],
    })

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(selected)} done")

# --- Analysis 4: Timing report ---
import numpy as np
print("\n=== Analysis 4: Inference Timing ===")
FULL_DATASET = 132062

for model_name, times in [("BLIP", blip_times), ("ViLT", vilt_times)]:
    if times:
        mean_t = np.mean(times)
        median_t = np.median(times)
        std_t = np.std(times)
        min_t = np.min(times)
        max_t = np.max(times)
        est_full = mean_t * FULL_DATASET / 3600
        print(f"\n{model_name}:")
        print(f"  Mean:     {mean_t:.3f}s")
        print(f"  Median:   {median_t:.3f}s")
        print(f"  Std:      {std_t:.3f}s")
        print(f"  Min:      {min_t:.3f}s")
        print(f"  Max:      {max_t:.3f}s")
        print(f"  Est. full run ({FULL_DATASET:,} qs): {est_full:.1f} hours")

print("\nPractical issues log:")
if practical_issues:
    for issue in practical_issues:
        print(f"  - {issue}")
else:
    print("  None encountered.")

# Save timing CSV
timing_rows = [{'model': 'BLIP', 'mean': round(np.mean(blip_times),4), 'median': round(np.median(blip_times),4),
                'std': round(np.std(blip_times),4), 'min': round(np.min(blip_times),4), 'max': round(np.max(blip_times),4),
                'n_samples': len(blip_times), 'est_full_hours': round(np.mean(blip_times)*FULL_DATASET/3600, 2)},
               {'model': 'ViLT', 'mean': round(np.mean(vilt_times),4), 'median': round(np.median(vilt_times),4),
                'std': round(np.std(vilt_times),4), 'min': round(np.min(vilt_times),4), 'max': round(np.max(vilt_times),4),
                'n_samples': len(vilt_times), 'est_full_hours': round(np.mean(vilt_times)*FULL_DATASET/3600, 2)}]
pd.DataFrame(timing_rows).to_csv(f"{OUT_DIR}/inference_timing.csv", index=False)
print(f"\nSaved inference_timing.csv")

# --- Analysis 5: Answer comparison ---
df = pd.DataFrame(results)
print("\n=== Analysis 5: Answer Comparison (first 20 rows) ===")
print(df[['question','gt_answer','blip_answer','vilt_answer','blip_exact_match','vilt_exact_match']].head(20).to_string(index=False))

blip_acc = df['blip_exact_match'].mean()
vilt_acc = df['vilt_exact_match'].mean()
print(f"\nRaw exact-match accuracy (case-insensitive):")
print(f"  BLIP: {blip_acc:.1%}")
print(f"  ViLT: {vilt_acc:.1%}")

# 5b. Format observations
print("\n=== 5b. Answer Format Observations ===")

# Case differences
case_mismatches_blip = sum(1 for r in results
    if r['blip_answer'].strip().lower() == r['gt_answer'].strip().lower()
    and r['blip_answer'].strip() != r['gt_answer'].strip())
case_mismatches_vilt = sum(1 for r in results
    if r['vilt_answer'].strip().lower() == r['gt_answer'].strip().lower()
    and r['vilt_answer'].strip() != r['gt_answer'].strip())
print(f"Case-only mismatches: BLIP={case_mismatches_blip}, ViLT={case_mismatches_vilt}")

# Multi-word answers
blip_multiword = sum(1 for r in results if len(r['blip_answer'].split()) > 1)
gqa_multiword = sum(1 for r in results if len(r['gt_answer'].split()) > 1)
print(f"Multi-word answers — BLIP: {blip_multiword}/{len(results)}, GQA gt: {gqa_multiword}/{len(results)}")

# ViLT vocab misses for wrong answers
vilt_wrong = [r for r in results if not r['vilt_exact_match']]
vilt_labels = set(vilt_model.config.id2label.values())
in_vocab_but_wrong = sum(1 for r in vilt_wrong if r['gt_answer'] in vilt_labels)
not_in_vocab = sum(1 for r in vilt_wrong if r['gt_answer'] not in vilt_labels)
print(f"ViLT wrong answers: {len(vilt_wrong)} total")
print(f"  Of those, gt IS in ViLT vocab: {in_vocab_but_wrong} (model error)")
print(f"  Of those, gt NOT in ViLT vocab: {not_in_vocab} (structural impossibility)")

# 5c. Normalization strategy
def normalize(s):
    s = s.strip().lower()
    # number words to digits
    num_map = {'zero':'0','one':'1','two':'2','three':'3','four':'4',
               'five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
    if s in num_map:
        s = num_map[s]
    # remove leading articles
    s = re.sub(r'^(a |an |the )', '', s)
    return s

blip_norm_acc = sum(1 for r in results if normalize(r['blip_answer']) == normalize(r['gt_answer'])) / len(results)
vilt_norm_acc = sum(1 for r in results if normalize(r['vilt_answer']) == normalize(r['gt_answer'])) / len(results)

print(f"\n=== 5c. Normalized exact-match accuracy ===")
print(f"  BLIP: {blip_norm_acc:.1%}  (was {blip_acc:.1%})")
print(f"  ViLT: {vilt_norm_acc:.1%}  (was {vilt_acc:.1%})")
print(f"\nNormalization strategy:")
print("  1. strip() and lower()")
print("  2. number words -> digits (one->1, two->2 ... ten->10)")
print("  3. remove leading articles (a/an/the)")

df.to_csv(f"{OUT_DIR}/answer_comparison.csv", index=False)
print(f"\nSaved answer_comparison.csv (all {len(df)} rows)")

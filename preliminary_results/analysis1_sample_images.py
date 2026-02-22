import zipfile
import random
import os

ZIP_PATH = "/Users/xuefeiyang/Documents/NLP/question-types-aware-vqa-analysis/data/images.zip"
OUT_DIR = "/Users/xuefeiyang/Documents/NLP/question-types-aware-vqa-analysis/preliminary_results/sample_images"
IDS_FILE = "/Users/xuefeiyang/Documents/NLP/question-types-aware-vqa-analysis/preliminary_results/sampled_image_ids.txt"

os.makedirs(OUT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH) as z:
    all_names = z.namelist()
    image_names = [n for n in all_names if n.lower().endswith('.jpg') and not n.endswith('/')]
    print(f"Total images in ZIP: {len(image_names)}")

    random.seed(42)
    sampled = random.sample(image_names, 100)

    for name in sampled:
        data = z.read(name)
        fname = os.path.basename(name)
        with open(os.path.join(OUT_DIR, fname), 'wb') as f:
            f.write(data)

    print(f"Extracted {len(sampled)} images to {OUT_DIR}")

image_ids = [os.path.splitext(os.path.basename(n))[0] for n in sampled]
with open(IDS_FILE, 'w') as f:
    f.write('\n'.join(image_ids))
print(f"Saved image IDs to {IDS_FILE}")
print("Sample IDs (first 5):", image_ids[:5])

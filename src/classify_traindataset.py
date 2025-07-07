import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === Paths ===
MODEL_PATH = r"D:/D-DRIVE/Assignment01/FMD_July5/models/mask_detector.h5"
INPUT_DIR = r"D:/D-DRIVE/Assignment01/FMD_July5/train_dataset"
OUTPUT_CSV = r"D:/D-DRIVE/Assignment01/FMD_July5/traindataset_labels.csv"

# === Load model ===
model = load_model(MODEL_PATH)
print(f" Loaded model: {MODEL_PATH}")

# === Valid extensions ===
valid_exts = ('.jpg', '.jpeg', '.png')

# === Initialize counters ===
mask_count = 1
nomask_count = 1

# === To store filename + label ===
results = []

# === Loop through images ===
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(valid_exts):
        continue

    img_path = os.path.join(INPUT_DIR, fname)

    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except Exception as e:
        print(f" Could not load {img_path}: {e}")
        continue

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_prob = model.predict(img_array, verbose=0)[0][0]
    pred_class = 'mask' if pred_prob >= 0.5 else 'nomask'

    if pred_class == 'mask':
        new_name = f"mask{mask_count}.jpg"
        mask_count += 1
    else:
        new_name = f"nomask{nomask_count}.jpg"
        nomask_count += 1

    new_path = os.path.join(INPUT_DIR, new_name)

    os.rename(img_path, new_path)

    print(f"{fname} → {pred_class} → {new_name}")

    # Save to results
    results.append({'filename': new_name, 'label': pred_class})

# === Save CSV ===
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n {mask_count-1} mask, {nomask_count-1} nomask")
print(f" CSV saved: {OUTPUT_CSV}")
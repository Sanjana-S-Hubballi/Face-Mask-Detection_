import os
import random
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === PATHS ===
MODEL_PATH = r"D:/D-DRIVE/Assignment01/FMD_July5/src/mobilenetv2_best.h5"
DATASET_DIR = r"D:/D-DRIVE/Assignment01/FMD_July5/train_dataset"
LABELS_CSV = r"D:/D-DRIVE/Assignment01/FMD_July5/traindataset_labels.csv"
RESULTS_DIR = r"D:/D-DRIVE/Assignment01/FMD_July5/results/train"

# === Create results folder if not exists ===
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Load labels ===
df = pd.read_csv(LABELS_CSV)
print(f" Loaded {len(df)} labels")

# === Load model ===
model = load_model(MODEL_PATH)
print(f" Loaded model: {MODEL_PATH}")

# === Sample N random images ===
N = 50
sample_df = df.sample(n=N).reset_index(drop=True)

y_true = []
y_pred = []

# === Loop through samples ===
for idx, row in sample_df.iterrows():
    fname = row['filename']
    label = row['label']

    # Add extension back
    img_path = os.path.join(DATASET_DIR, f"{fname}")  # adjust if .png

    if not os.path.exists(img_path):
        print(f" File not found: {img_path}")
        continue

    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except Exception as e:
        print(f" Could not load {img_path}: {e}")
        continue

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_prob = model.predict(img_array, verbose=0)[0][0]
    pred_class = 'nomask' if pred_prob >= 0.5 else 'mask'

    y_true.append(label)
    y_pred.append(pred_class)

    print(f"{fname} → True: {label} → Predicted: {pred_class} ({pred_prob:.4f})")

# === Compute Confusion Matrix ===
labels = ['mask', 'nomask']
cm = confusion_matrix(y_true, y_pred, labels=labels)
print("\n Confusion Matrix:\n", cm)

# === Plot Confusion Matrix ===
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
plt.title("Confusion Matrix - Random Sample")

# === Save confusion matrix ===
conf_matrix_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
plt.savefig(conf_matrix_path)
plt.show()
print(f" Confusion Matrix saved: {conf_matrix_path}")

# === Compute FAR & FRR ===
TP = cm[0][0]  # mask predicted as mask
FN = cm[0][1]  # mask predicted as nomask
FP = cm[1][0]  # nomask predicted as mask
TN = cm[1][1]  # nomask predicted as nomask

FAR = FP / (FP + TN) if (FP + TN) > 0 else 0
FRR = FN / (TP + FN) if (TP + FN) > 0 else 0

print(f"\n FAR (False Accept Rate): {FAR:.4f}")
print(f" FRR (False Reject Rate): {FRR:.4f}")

# === Save predictions ===
results_df = pd.DataFrame({
    'filename': sample_df['filename'],
    'true_label': y_true,
    'predicted_label': y_pred
})
predictions_csv_path = os.path.join(RESULTS_DIR, 'sample_evaluation.csv')
results_df.to_csv(predictions_csv_path, index=False)
print(f" Predictions saved: {predictions_csv_path}")

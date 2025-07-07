import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Paths ===
MODEL_PATH = r"D:/D-DRIVE/Assignment01/FMD_July5/src/mobilenetv2_best.h5"
DETECTED_FACES_DIR = r"D:/D-DRIVE/Assignment01/FMD_July5/detected_faces"
GROUND_TRUTH_CSV = r"D:/D-DRIVE/Assignment01/FMD_July5/labels.csv"
OUTPUT_CSV = r"D:/D-DRIVE/Assignment01/FMD_July5/results/test/detected_faces_predictions.csv"
CONFUSION_MATRIX_IMG = r"D:/D-DRIVE/Assignment01/FMD_July5/results/test/confusion_matrix.png"
RESULTS_TXT = r"D:/D-DRIVE/Assignment01/FMD_July5/results/test/results.txt"

# === Load model ===
model = load_model(MODEL_PATH)
print(f" Loaded model: {MODEL_PATH}")

# === Load ground truth ===
ground_truth_df = pd.read_csv(GROUND_TRUTH_CSV)

# === Valid extensions ===
valid_exts = ('.jpg', '.jpeg', '.png')

# === Store results ===
results = []

# === Loop through detected faces ===
for fname in os.listdir(DETECTED_FACES_DIR):
    if not fname.lower().endswith(valid_exts):
        continue

    img_path = os.path.join(DETECTED_FACES_DIR, fname)

    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except Exception as e:
        print(f" Could not load {img_path}: {e}")
        continue

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_prob = model.predict(img_array, verbose=0)[0][0]
    pred_class = 'nomask' if pred_prob >= 0.5 else 'mask'

    # === Get true label ===
    true_label_row = ground_truth_df[ground_truth_df['filename'] == fname]
    if true_label_row.empty:
        print(f" No ground truth for {fname}")
        continue
    true_label = true_label_row['true_label'].values[0]

    print(f"{fname} → True: {true_label} → Predicted: {pred_class} ({pred_prob:.4f})")

    results.append({
        'filename': fname,
        'true_label': true_label,
        'predicted_label': pred_class,
        'probability': f"{pred_prob:.4f}"
    })

# === Save predictions to CSV ===
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n Predictions saved: {OUTPUT_CSV}")

# === Compute confusion matrix ===
y_true = df['true_label']
y_pred = df['predicted_label']
cm = confusion_matrix(y_true, y_pred, labels=['mask', 'nomask'])
print("\nConfusion Matrix:\n", cm)

# === Plot confusion matrix ===
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['mask', 'nomask'])
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - Detected Faces")
plt.savefig(CONFUSION_MATRIX_IMG)
plt.close()
print(f" Confusion matrix saved: {CONFUSION_MATRIX_IMG}")

# === Compute FAR & FRR ===
TP = cm[0][0]   # mask predicted mask
FN = cm[0][1]   # mask predicted nomask
FP = cm[1][0]   # nomask predicted mask
TN = cm[1][1]   # nomask predicted nomask

FAR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Accept Rate
FRR = FN / (TP + FN) if (TP + FN) > 0 else 0  # False Reject Rate

print(f"\n FAR (False Accept Rate): {FAR:.4f}")
print(f" FRR (False Reject Rate): {FRR:.4f}")

# === Save FAR, FRR to text ===
os.makedirs(os.path.dirname(RESULTS_TXT), exist_ok=True)
with open(RESULTS_TXT, 'w') as f:
    f.write(f"Confusion Matrix:\n{cm}\n\n")
    f.write(f"FAR (False Accept Rate): {FAR:.4f}\n")
    f.write(f"FRR (False Reject Rate): {FRR:.4f}\n")

print(f" Results saved: {RESULTS_TXT}")

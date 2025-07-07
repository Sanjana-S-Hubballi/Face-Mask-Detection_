import cv2
import os

# === PATHS ===
prototxt_path = "D:/D-DRIVE/Assignment01/FMD_July5/models/deploy.prototxt"
weights_path = "D:/D-DRIVE/Assignment01/FMD_July5/models/res10_300x300_ssd_iter_140000.caffemodel"

input_folder = "D:/D-DRIVE/Assignment01/FMD_July5/dataset"
output_folder = "D:/D-DRIVE/Assignment01/FMD_July5/detected_faces"

os.makedirs(output_folder, exist_ok=True)

# === Load ResNet10 SSD model ===
net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

# === Fixed output size ===
OUTPUT_SIZE = (224, 224)

count = 0

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = image[startY:endY, startX:endX]

            if face.size == 0:
                continue

            # Resize face crop to fixed size before saving
            face_resized = cv2.resize(face, OUTPUT_SIZE)

            save_name = f"face_{count}.jpg"
            cv2.imwrite(os.path.join(output_folder, save_name), face_resized)
            count += 1

print(f"Done! Detected faces saved in {output_folder}")
import os
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# === Paths ===
CSV_PATH = 'D:/D-DRIVE/Assignment01/FMD_July5/traindataset_labels.csv'
IMAGES_DIR = 'D:/D-DRIVE/Assignment01/FMD_July5/train_dataset/'

# === Load CSV ===
df = pd.read_csv(CSV_PATH)
print("\n CSV Loaded:")
print(df.head())

# === Clean labels ===
df['label'] = df['label'].astype(str).str.strip().str.lower()
df['filename'] = df['filename'].astype(str).str.strip()

# === Fix extensions ===
valid_exts = ('.jpg', '.jpeg', '.png')
df.loc[~df['filename'].str.lower().str.endswith(valid_exts), 'filename'] += '.jpg'

print("\n Cleaned labels and filenames:")
print(df.head())
print("Labels:", df['label'].unique())

# === Check files in images directory ===
print("\n Example files in images folder:", os.listdir(IMAGES_DIR)[:5])

# === ImageDataGenerator with strong augmentation ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

# === Flow from dataframe ===
train_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMAGES_DIR,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMAGES_DIR,
    x_col='filename',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

print("\n Training samples:", train_gen.samples)
print(" Validation samples:", val_gen.samples)
print(" Classes:", train_gen.class_indices)

# === Build MobileNetV2 ===
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === Freeze base layers ===
for layer in base_model.layers:
    layer.trainable = False

# === Compile ===
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Callbacks ===
earlystop = EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=4,
    verbose=1,
    min_lr=1e-7
)

checkpoint = ModelCheckpoint(
    'mobilenetv2_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# === Train top layers ===
print("\n Starting initial training...")
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[earlystop, reduce_lr, checkpoint]
)

# === Unfreeze more layers for deeper fine-tuning ===
for layer in base_model.layers[-80:]:
    layer.trainable = True

# === Re-compile for fine-tuning ===
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Fine-tune ===
print("\n Starting fine-tuning...")
history_finetune = model.fit(
    train_gen,
    epochs=70,
    validation_data=val_gen,
    callbacks=[earlystop, reduce_lr, checkpoint]
)

# === Save final model ===
model.save('mobilenetv2_model.h5')
print("\n Final model saved as mobilenetv2_model.h5")
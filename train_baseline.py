"""
Baseline transfer learning training script
- Model: EfficientNetB0 pretrained on ImageNet
- Workflow: train head (backbone frozen) -> unfreeze last block -> fine-tune
- Uses CSVs: datasets/ham10000/train.csv, val.csv, test.csv
- Saves: saved_model/best.h5 and saved_model/final.h5, TensorBoard logs in logs/
"""

import os
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------
# Args / Config
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train_csv', type=str, default='datasets/ham10000/train.csv')
parser.add_argument('--val_csv',   type=str, default='datasets/ham10000/val.csv')
parser.add_argument('--test_csv',  type=str, default='datasets/ham10000/test.csv')
parser.add_argument('--img_size',  type=int, default=224)
parser.add_argument('--batch',     type=int, default=32)
parser.add_argument('--epochs_head', type=int, default=8)
parser.add_argument('--epochs_ft',   type=int, default=12)
parser.add_argument('--lr_head',   type=float, default=1e-3)
parser.add_argument('--lr_ft',     type=float, default=1e-5)
parser.add_argument('--dropout',   type=float, default=0.3)
parser.add_argument('--savedir',   type=str, default='saved_model')
parser.add_argument('--tensorboard', action='store_true', help='enable tensorboard logs')
parser.add_argument('--mixed_precision', action='store_true', help='enable mixed precision (fp16)')
args = parser.parse_args()

os.makedirs(args.savedir, exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ---------------------------
# Mixed precision (optional)
# ---------------------------
if args.mixed_precision:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Enabled mixed precision (mixed_float16).")
    except Exception as e:
        print("Could not enable mixed precision:", e)

# ---------------------------
# Read CSVs
# ---------------------------
train_df = pd.read_csv(args.train_csv)
val_df   = pd.read_csv(args.val_csv)
test_df  = pd.read_csv(args.test_csv) if os.path.exists(args.test_csv) else None

# Expect CSV columns: image_path, label (0..6), dx, image_id
print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

NUM_CLASSES = len(train_df['label'].unique())
IMG_SIZE = args.img_size
BATCH = args.batch
AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------
# Helper: augmentation & parsing
# ---------------------------
def random_augment(image):
    # simple, fast augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # small rotations
    angle = tf.random.uniform([], -0.1, 0.1)  # radians ~ ±5.7 degrees
    image = tfa.image.rotate(image, angles=angle) if 'tfa' in globals() else image
    # random brightness/contrast
    image = tf.image.random_brightness(image, 0.08)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image

def parse_image(path, label, augment=False):
    # path is a string like "E:/skin-project/datasets/ham10000/images/ISIC_..."
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # handles jpg/jpeg
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0  # scale to [0,1]
    if augment:
        image = random_augment(image)
    return image, tf.one_hot(label, NUM_CLASSES)

# ---------------------------
# Build tf.data datasets
# ---------------------------
def df_to_dataset(df, batch=BATCH, shuffle=True, augment=False):
    paths = df['image_path'].astype(str).values
    labels = df['label'].astype(int).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 2048))
    ds = ds.map(lambda p, l: tf.py_function(func=lambda pp, ll: parse_image(pp, ll, augment),
                                            inp=[p, l],
                                            Tout=(tf.float32, tf.float32)),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds

# Try to import tensorflor_addons for rotate; optional
try:
    import tensorflow_addons as tfa
    print("tensorflow_addons available -> using rotate augmentation.")
except Exception:
    tfa = None

train_ds = df_to_dataset(train_df, batch=BATCH, shuffle=True, augment=True)
val_ds   = df_to_dataset(val_df, batch=BATCH, shuffle=False, augment=False)
test_ds  = df_to_dataset(test_df, batch=BATCH, shuffle=False, augment=False) if test_df is not None else None

# ---------------------------
# Compute class weights (helpful for imbalance)
# ---------------------------
classes = train_df['label'].values
class_weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
class_weight_dict = {i: float(w) for i, w in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

# ---------------------------
# Build model (EfficientNetB0 backbone)
# ---------------------------
base = tf.keras.applications.EfficientNetB0(include_top=False,
                                            input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                            weights='imagenet',
                                            pooling='avg')
base.trainable = False  # freeze for head training

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.Dropout(args.dropout)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)  # ensure float32 for output when mixed precision used
model = models.Model(inputs, outputs)

model.summary()

# ---------------------------
# Compile for head training
# ---------------------------
optimizer = optimizers.Adam(learning_rate=args.lr_head)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# ---------------------------
# Callbacks
# ---------------------------
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
tb_logdir = f"logs/run-{timestamp}"
callbacks_list = [
    callbacks.ModelCheckpoint(os.path.join(args.savedir, 'best_head.h5'),
                              save_best_only=True, monitor='val_auc', mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
]
if args.tensorboard:
    callbacks_list.append(callbacks.TensorBoard(log_dir=tb_logdir))

# ---------------------------
# Train head
# ---------------------------
print(">>> Training head (backbone frozen)")
history_head = model.fit(train_ds,
                         validation_data=val_ds,
                         epochs=args.epochs_head,
                         class_weight=class_weight_dict,
                         callbacks=callbacks_list)

# ---------------------------
# Fine-tune: unfreeze last layers
# ---------------------------
print(">>> Fine-tuning: unfreeze last layers of backbone")
# Unfreeze the entire base or just last N layers. We'll unfreeze last 30 layers.
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

# Lower LR for fine-tuning
optimizer_ft = optimizers.Adam(learning_rate=args.lr_ft)
model.compile(optimizer=optimizer_ft,
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

callbacks_ft = [
    callbacks.ModelCheckpoint(os.path.join(args.savedir, 'best_finetune.h5'),
                              save_best_only=True, monitor='val_auc', mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
]
if args.tensorboard:
    callbacks_ft.append(callbacks.TensorBoard(log_dir=tb_logdir + "-ft"))

history_ft = model.fit(train_ds,
                       validation_data=val_ds,
                       epochs=args.epochs_ft,
                       class_weight=class_weight_dict,
                       callbacks=callbacks_ft)

# Save final
model.save(os.path.join(args.savedir, 'final_model'))

# ---------------------------
# Evaluate on test set (optional)
# ---------------------------
if test_ds is not None:
    print(">>> Evaluating on test set")
    results = model.evaluate(test_ds)
    print("Test results (loss, acc, auc):", results)
else:
    print("No test CSV / test dataset found. Skipping test eval.")

print("Training complete. Best models saved in", args.savedir)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler

DATA_DIR  = r"C:\Users\bhask\OneDrive\Desktop\HardwareTrojanDetection\data"
TRACE_LEN = 1000
MAX_FILES = 40000   # 20k clean + 20k trojan
os.makedirs("results", exist_ok=True)

# ---- Find files ----
def find_trace_files(data_dir):
    out = []
    for chip in os.listdir(data_dir):
        chip_path = os.path.join(data_dir, chip)
        if not os.path.isdir(chip_path): continue
        for sub in os.listdir(chip_path):
            sub_path = os.path.join(chip_path, sub)
            if not os.path.isdir(sub_path): continue
            if   'TrojanDisabled'  in sub: label = 0
            elif 'TrojanEnabled'   in sub: label = 1
            elif 'TrojanTriggered' in sub: label = 1
            else: continue
            for inner in os.listdir(sub_path):
                inner_path = os.path.join(sub_path, inner)
                if os.path.isdir(inner_path):
                    for fname in os.listdir(inner_path):
                        if fname.endswith('.csv'):
                            out.append({'filepath': os.path.join(inner_path, fname),
                                        'label': label, 'chip': chip})
                elif inner.endswith('.csv'):
                    out.append({'filepath': inner_path, 'label': label, 'chip': chip})
    return out

print("Scanning files...")
all_files = find_trace_files(DATA_DIR)
clean_files  = [f for f in all_files if f['label'] == 0]
trojan_files = [f for f in all_files if f['label'] == 1]
print(f"Clean: {len(clean_files)}  |  Trojan: {len(trojan_files)}")

n_each = MAX_FILES // 2
random.seed(42)
selected = (random.sample(clean_files,  n_each) +
            random.sample(trojan_files, n_each))
random.shuffle(selected)
print(f"Using {len(selected)} traces ({n_each} clean + {n_each} trojan)")

# ---- Peek at raw values to understand scale ----
print("\nPeeking at raw values...")
sample_trace = pd.read_csv(selected[0]['filepath'], header=None).values.flatten().astype(np.float32)
print(f"  Raw min={sample_trace.min():.4f}  max={sample_trace.max():.4f}"
      f"  mean={sample_trace.mean():.4f}  std={sample_trace.std():.4f}")

# ---- Load all traces ----
print("\nLoading traces...")
X_list, y_list = [], []
for i, f in enumerate(selected):
    if i % 5000 == 0: print(f"  {i}/{len(selected)}...")
    try:
        t = pd.read_csv(f['filepath'], header=None).values.flatten().astype(np.float32)
        if len(t) < TRACE_LEN: continue
        t = t[:TRACE_LEN]
        X_list.append(t)
        y_list.append(f['label'])
    except: pass

X_raw = np.array(X_list, dtype=np.float32)   # shape: (N, 1000)
y     = np.array(y_list,  dtype=np.float32)
print(f"Loaded: {X_raw.shape}  |  Clean: {(y==0).sum()}  Trojan: {(y==1).sum()}")

# ---- Normalize using global stats (NOT per-trace) ----
# Per-trace normalization removes the signal difference between clean & trojan!
print("\nNormalizing (global z-score)...")
global_mean = X_raw.mean()
global_std  = X_raw.std()
print(f"  Global mean={global_mean:.4f}  std={global_std:.4f}")
X_norm = (X_raw - global_mean) / (global_std + 1e-8)

# Clip extreme outliers
X_norm = np.clip(X_norm, -5, 5)
print(f"  After norm — min={X_norm.min():.3f}  max={X_norm.max():.3f}")

# Save normalization params for later use
np.save("results/cnn_norm_mean.npy", global_mean)
np.save("results/cnn_norm_std.npy",  global_std)

# Reshape for CNN: (N, timesteps, channels)
X = X_norm[..., np.newaxis]

# ---- Split ----
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")

# Verify label balance
print(f"Train — Clean: {(y_tr==0).sum()}  Trojan: {(y_tr==1).sum()}")
print(f"Test  — Clean: {(y_te==0).sum()}  Trojan: {(y_te==1).sum()}")

# ---- Quick sanity check: are clean & trojan traces actually different? ----
clean_mean  = X_raw[y==0].mean(axis=0)
trojan_mean = X_raw[y==1].mean(axis=0)
diff = np.abs(clean_mean - trojan_mean).mean()
print(f"\nMean absolute difference between clean & trojan traces: {diff:.6f}")
if diff < 1e-6:
    print("WARNING: Clean and Trojan traces look identical! Check your data labels.")
else:
    print("Good: traces are distinguishable.")

plt.figure(figsize=(12, 4))
plt.plot(clean_mean[:500],  label='Clean mean',  alpha=0.8, linewidth=1)
plt.plot(trojan_mean[:500], label='Trojan mean', alpha=0.8, linewidth=1)
plt.title("Average Power Trace: Clean vs Trojan (first 500 samples)")
plt.xlabel("Sample"); plt.ylabel("Power")
plt.legend(); plt.tight_layout()
plt.savefig("results/clean_vs_trojan_mean.png", dpi=150)
plt.show()
print("Saved: results/clean_vs_trojan_mean.png")

# ---- Build CNN ----
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

print(f"\nTensorFlow version: {tf.__version__}")

def build_cnn(length):
    inp = layers.Input(shape=(length, 1))

    # Block 1 — wide kernel to catch low-freq patterns
    x = layers.Conv1D(64, kernel_size=11, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(4)(x)       # 1000 → 250

    # Block 2
    x = layers.Conv1D(128, kernel_size=7, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(4)(x)       # 250 → 62

    # Block 3
    x = layers.Conv1D(256, kernel_size=5, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Head
    x = layers.Dense(128, activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inp, out)

model = build_cnn(TRACE_LEN)
model.summary()

# Use a lower learning rate + label smoothing to avoid collapse
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

cb = [
    callbacks.EarlyStopping(monitor='val_auc', patience=10,
                            restore_best_weights=True, verbose=1,
                            mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                patience=4, verbose=1, mode='max'),
    callbacks.ModelCheckpoint("results/cnn_best.keras",
                              monitor='val_auc', save_best_only=True,
                              verbose=1, mode='max')
]

print("\nTraining CNN...")
history = model.fit(
    X_tr, y_tr,
    validation_split=0.2,
    epochs=50,
    batch_size=128,    # larger batch = more stable gradients
    callbacks=cb,
    verbose=1
)

# ---- Evaluate ----
print("\nEvaluating...")
y_prob = model.predict(X_te, batch_size=128).flatten()
y_pred = (y_prob > 0.5).astype(int)

print("\n" + "="*50)
print("  1D-CNN Results")
print("="*50)
print(classification_report(y_te.astype(int), y_pred,
                             target_names=['Clean', 'Trojan'],
                             zero_division=0))

cm = confusion_matrix(y_te.astype(int), y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Clean', 'Trojan'],
            yticklabels=['Clean', 'Trojan'], ax=ax)
ax.set_title("1D-CNN — Confusion Matrix")
ax.set_ylabel("True"); ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("results/cm_CNN.png", dpi=150)
plt.show()

# Training curves
metrics_to_plot = [('loss','val_loss'), ('accuracy','val_accuracy'),
                   ('auc','val_auc')]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ['Loss', 'Accuracy', 'AUC']
for ax, (tr_m, val_m), title in zip(axes, metrics_to_plot, titles):
    if tr_m in history.history:
        ax.plot(history.history[tr_m],  label='Train')
        ax.plot(history.history[val_m], label='Val')
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/cnn_training_curves.png", dpi=150)
plt.show()

model.save("results/cnn_model.keras")
print("\nDone!")
print("Saved: results/cnn_model.keras")
print("Saved: results/cm_CNN.png")
print("Saved: results/cnn_training_curves.png")
print("Saved: results/clean_vs_trojan_mean.png")
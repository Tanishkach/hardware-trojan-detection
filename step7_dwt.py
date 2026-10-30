import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import pickle
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
import shap

DATA_DIR = r"C:\Users\bhask\OneDrive\Desktop\HardwareTrojanDetection\data"
os.makedirs("results", exist_ok=True)

def find_trace_files(data_dir):
    out = []
    for chip in os.listdir(data_dir):
        chip_path = os.path.join(data_dir, chip)
        if not os.path.isdir(chip_path): continue
        for sub in os.listdir(chip_path):
            sub_path = os.path.join(chip_path, sub)
            if not os.path.isdir(sub_path): continue
            if   'TrojanDisabled'  in sub: label = 'clean'
            elif 'TrojanEnabled'   in sub: label = 'trojan'
            elif 'TrojanTriggered' in sub: label = 'trojan'
            else: continue
            for inner in os.listdir(sub_path):
                inner_path = os.path.join(sub_path, inner)
                if os.path.isdir(inner_path):
                    for fname in os.listdir(inner_path):
                        if fname.endswith('.csv'):
                            out.append({'filepath': os.path.join(inner_path, fname),
                                        'label': label, 'chip': chip, 'subfolder': sub})
                elif inner.endswith('.csv'):
                    out.append({'filepath': inner_path,
                                'label': label, 'chip': chip, 'subfolder': sub})
    return out

def load_trace(filepath):
    return pd.read_csv(filepath, header=None).values.flatten().astype(np.float64)

# 2. FEATURE EXTRACTION

WAVELET  = 'db4'
LEVEL    = 4
# subband names: cA (approximation) + cD4, cD3, cD2, cD1 (detail, coarse→fine)
SUBBAND_NAMES = ['cA', 'cD4', 'cD3', 'cD2', 'cD1']

def dwt_subband_features(coeffs, name):
    """6 stats for one DWT subband."""
    c = np.array(coeffs)
    return {
        f'{name}_mean':     np.mean(c),
        f'{name}_std':      np.std(c),
        f'{name}_energy':   np.sum(c**2),
        f'{name}_max_abs':  np.max(np.abs(c)),
        f'{name}_mean_abs': np.mean(np.abs(c)),
        f'{name}_rms':      np.sqrt(np.mean(c**2)),
    }

def extract_dwt_features(trace):
    trace = trace - np.mean(trace)          # remove DC offset

    # ---- Time-domain features (same 6 as FFT file) ----
    time_feats = {
        'mean':         np.mean(trace),
        'rms':          np.sqrt(np.mean(trace**2)),
        'std':          np.std(trace),
        'skewness':     skew(trace),
        'kurtosis':     kurtosis(trace),
        'peak_to_peak': np.max(trace) - np.min(trace),
    }

    # DWT features
    # wavedec returns [cA, cD_level, ..., cD1]  (coarse → fine)
    coeffs = pywt.wavedec(trace, wavelet=WAVELET, level=LEVEL)
    # coeffs[0]=cA, coeffs[1]=cD4, coeffs[2]=cD3, coeffs[3]=cD2, coeffs[4]=cD1
    dwt_feats = {}
    for name, c in zip(SUBBAND_NAMES, coeffs):
        dwt_feats.update(dwt_subband_features(c, name))

    return {**time_feats, **dwt_feats}

# 3. LOAD & PROCESS ALL TRACES

print("Scanning files...")
all_files = find_trace_files(DATA_DIR)
print(f"Total files: {len(all_files)}")

print("Extracting DWT features (this will take a while)...")
rows = []
for f in tqdm(all_files):
    try:
        trace = load_trace(f['filepath'])
        if len(trace) < 100: continue
        feats = extract_dwt_features(trace)
        rows.append({'chip': f['chip'], 'subfolder': f['subfolder'],
                     'label': f['label'], **feats})
    except Exception as e:
        pass

df = pd.DataFrame(rows)
df.to_csv("results/dwt_features.csv", index=False)
print(f"\nDWT feature matrix: {df.shape}")
print("Label distribution:")
print(df['label'].value_counts())

# 4. PREPARE X, y
feature_cols = [c for c in df.columns if c not in ['chip','subfolder','label']]
print(f"\nTotal DWT features: {len(feature_cols)}")
print("Feature names:", feature_cols)

X = df[feature_cols].fillna(0).values
y = (df['label'] == 'trojan').astype(int).values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
with open("results/dwt_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {len(X_tr)}  |  Test: {len(X_te)}")
print(f"Train — Clean: {(y_tr==0).sum()}  Trojan: {(y_tr==1).sum()}")
print(f"Test  — Clean: {(y_te==0).sum()}  Trojan: {(y_te==1).sum()}")

# 5. HELPER — save confusion matrix
def save_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean','Trojan'],
                yticklabels=['Clean','Trojan'], ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150)
    plt.close()
    print(f"Saved: results/{filename}")


# 6. MODEL A — RANDOM FOREST

print("\n" + "="*55)
print("  RANDOM FOREST (DWT features)")
print("="*55)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
y_pred_rf = rf.predict(X_te)

rf_acc = accuracy_score(y_te, y_pred_rf)
rf_f1  = f1_score(y_te, y_pred_rf, average='weighted')
rf_report = classification_report(y_te, y_pred_rf,
                                   target_names=['Clean','Trojan'],
                                   zero_division=0)
print(rf_report)
print(f"Accuracy: {rf_acc:.4f}  |  F1: {rf_f1:.4f}")

with open("results/dwt_rf_classification_report.txt", "w") as f:
    f.write("DWT + Random Forest Classification Report\n")
    f.write("="*45 + "\n")
    f.write(rf_report)
    f.write(f"\nAccuracy: {rf_acc:.4f}  |  F1: {rf_f1:.4f}\n")
print("Saved: results/dwt_rf_classification_report.txt")

save_cm(y_te, y_pred_rf,
        "DWT + Random Forest — Confusion Matrix",
        "dwt_rf_confusion_matrix.png")

with open("results/dwt_rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

# RF feature importance
imp = pd.DataFrame({'Feature': feature_cols,
                    'Importance': rf.feature_importances_})
imp = imp.sort_values('Importance', ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=imp, x='Importance', y='Feature', palette='viridis', ax=ax)
ax.set_title("Top 20 DWT Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("results/dwt_rf_feature_importance.png", dpi=150)
plt.close()
print("Saved: results/dwt_rf_feature_importance.png")


# 7. MODEL B — SVM (subsample for speed)

print("\n" + "="*55)
print("  SVM RBF (DWT features, 20k subsample)")
print("="*55)
rng = np.random.RandomState(42)
svm_idx = rng.choice(len(X_sc), 20000, replace=False)
X_sv = X_sc[svm_idx]; y_sv = y[svm_idx]
X_tr_sv, X_te_sv, y_tr_sv, y_te_sv = train_test_split(
    X_sv, y_sv, test_size=0.2, random_state=42, stratify=y_sv)

svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
print("Training SVM...")
svm.fit(X_tr_sv, y_tr_sv)
y_pred_svm = svm.predict(X_te_sv)

svm_acc = accuracy_score(y_te_sv, y_pred_svm)
svm_f1  = f1_score(y_te_sv, y_pred_svm, average='weighted')
svm_report = classification_report(y_te_sv, y_pred_svm,
                                    target_names=['Clean','Trojan'],
                                    zero_division=0)
print(svm_report)
print(f"Accuracy: {svm_acc:.4f}  |  F1: {svm_f1:.4f}")

with open("results/dwt_svm_classification_report.txt", "w") as f:
    f.write("DWT + SVM Classification Report\n")
    f.write("="*45 + "\n")
    f.write(svm_report)
    f.write(f"\nAccuracy: {svm_acc:.4f}  |  F1: {svm_f1:.4f}\n")
print("Saved: results/dwt_svm_classification_report.txt")

save_cm(y_te_sv, y_pred_svm,
        "DWT + SVM RBF — Confusion Matrix (20k subsample)",
        "dwt_svm_confusion_matrix.png")

with open("results/dwt_svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)


# 8. MODEL C — DEEP MLP ON DWT FEATURES

print("\n" + "="*55)
print("  DEEP MLP (DWT features)")
print("="*55)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

def build_mlp(n_features):
    inp = layers.Input(shape=(n_features,))
    x = layers.Dense(512, kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out)

mlp = build_mlp(X_sc.shape[1])
mlp.summary()
mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

cb = [
    callbacks.EarlyStopping(monitor='val_auc', patience=10,
                            restore_best_weights=True, verbose=1, mode='max'),
    callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                patience=4, verbose=1, mode='max'),
    callbacks.ModelCheckpoint("results/dwt_mlp_best.keras",
                              monitor='val_auc', save_best_only=True,
                              verbose=1, mode='max')
]

history = mlp.fit(
    X_tr, y_tr,
    validation_split=0.2,
    epochs=100,
    batch_size=512,
    callbacks=cb,
    verbose=1
)

y_prob_mlp = mlp.predict(X_te, batch_size=512).flatten()
y_pred_mlp = (y_prob_mlp > 0.5).astype(int)

mlp_acc = accuracy_score(y_te, y_pred_mlp)
mlp_f1  = f1_score(y_te, y_pred_mlp, average='weighted')
mlp_report = classification_report(y_te, y_pred_mlp,
                                    target_names=['Clean','Trojan'],
                                    zero_division=0)
print(mlp_report)
print(f"Accuracy: {mlp_acc:.4f}  |  F1: {mlp_f1:.4f}")

with open("results/dwt_mlp_classification_report.txt", "w") as f:
    f.write("DWT + Deep MLP Classification Report\n")
    f.write("="*45 + "\n")
    f.write(mlp_report)
    f.write(f"\nAccuracy: {mlp_acc:.4f}  |  F1: {mlp_f1:.4f}\n")
print("Saved: results/dwt_mlp_classification_report.txt")

save_cm(y_te, y_pred_mlp,
        "DWT + Deep MLP — Confusion Matrix",
        "dwt_mlp_confusion_matrix.png")

mlp.save("results/dwt_mlp_model.keras")

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (tr_k, val_k), title in zip(
        axes,
        [('loss','val_loss'),('accuracy','val_accuracy'),('auc','val_auc')],
        ['Loss','Accuracy','AUC']):
    ax.plot(history.history[tr_k],  label='Train')
    ax.plot(history.history[val_k], label='Val')
    ax.set_title(f"DWT MLP — {title}")
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/dwt_mlp_training_curves.png", dpi=150)
plt.close()
print("Saved: results/dwt_mlp_training_curves.png")


# 9. SHAP ON RANDOM FOREST

print("\nRunning SHAP on DWT Random Forest...")
rng2  = np.random.RandomState(42)
s_idx = rng2.choice(len(X_sc), 100, replace=False)
X_sh  = X_sc[s_idx]
y_sh  = y[s_idx]

explainer   = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
shap_raw    = explainer.shap_values(X_sh, check_additivity=False)
arr         = np.array(shap_raw)
sv          = arr[:, :, 1] if arr.ndim == 3 else (
              np.array(shap_raw[1]) if isinstance(shap_raw, list) else arr)
print(f"SHAP sv shape: {sv.shape}")

ev       = explainer.expected_value
base_val = float(ev[1]) if hasattr(ev, "__len__") else float(ev)

# SHAP bar
fig, ax = plt.subplots(figsize=(10, 7))
mean_shap = np.abs(sv).mean(axis=0)
top_idx   = np.argsort(mean_shap)[-20:]
ax.barh([feature_cols[i] for i in top_idx], mean_shap[top_idx], color='steelblue')
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("DWT SHAP Feature Importance — Trojan Detection (RF)")
plt.tight_layout()
plt.savefig("results/dwt_shap_bar.png", dpi=150)
plt.close()
print("Saved: results/dwt_shap_bar.png")

# SHAP beeswarm
fig2, ax2 = plt.subplots(figsize=(12, 8))
scatter_x = sv[:, top_idx].T
feat_vals  = X_sh[:, top_idx].T
for i, (rx, rv) in enumerate(zip(scatter_x, feat_vals)):
    colors = ["#d73027" if v > 0 else "#4575b4" for v in rv]
    ax2.scatter(rx, [i]*len(rx), c=colors, alpha=0.6, s=20)
ax2.set_yticks(range(len(top_idx)))
ax2.set_yticklabels([feature_cols[i] for i in top_idx])
ax2.axvline(0, color='black', linewidth=0.8)
ax2.set_xlabel("SHAP value (impact on Trojan prediction)")
ax2.set_title("DWT SHAP Summary — Feature Impact on Trojan Detection")
plt.tight_layout()
plt.savefig("results/dwt_shap_summary.png", dpi=150)
plt.close()
print("Saved: results/dwt_shap_summary.png")

# Waterfall — Trojan sample
t_idx = np.where(y_sh == 1)[0][0]
sv1   = sv[t_idx]
order = np.argsort(np.abs(sv1))[-15:]
fig3, ax3 = plt.subplots(figsize=(10, 7))
colors3 = ["#d73027" if v > 0 else "#4575b4" for v in sv1[order]]
ax3.barh([feature_cols[i] for i in order], sv1[order], color=colors3)
ax3.axvline(0, color='black', linewidth=0.8)
ax3.set_xlabel("SHAP value")
ax3.set_title("DWT SHAP Waterfall — Single Trojan Sample")
plt.tight_layout()
plt.savefig("results/dwt_shap_waterfall_trojan.png", dpi=150)
plt.close()
print("Saved: results/dwt_shap_waterfall_trojan.png")

# Waterfall — Clean sample
c_idx = np.where(y_sh == 0)[0][0]
sv0   = sv[c_idx]
order0 = np.argsort(np.abs(sv0))[-15:]
fig4, ax4 = plt.subplots(figsize=(10, 7))
colors4 = ["#d73027" if v > 0 else "#4575b4" for v in sv0[order0]]
ax4.barh([feature_cols[i] for i in order0], sv0[order0], color=colors4)
ax4.axvline(0, color='black', linewidth=0.8)
ax4.set_xlabel("SHAP value")
ax4.set_title("DWT SHAP Waterfall — Single Clean Sample")
plt.tight_layout()
plt.savefig("results/dwt_shap_waterfall_clean.png", dpi=150)
plt.close()
print("Saved: results/dwt_shap_waterfall_clean.png")

# 10. FINAL SUMMARY

print("\n" + "="*55)
print("  DWT RESULTS SUMMARY")
print("="*55)
print(f"  Random Forest  — Accuracy: {rf_acc:.4f}  |  F1: {rf_f1:.4f}  (full dataset)")
print(f"  SVM (RBF)      — Accuracy: {svm_acc:.4f}  |  F1: {svm_f1:.4f}  (20k subsample)")
print(f"  Deep MLP       — Accuracy: {mlp_acc:.4f}  |  F1: {mlp_f1:.4f}  (full dataset)")

print("\nTop 10 DWT features by SHAP importance:")
top10 = sorted(zip(feature_cols, mean_shap), key=lambda x: -x[1])[:10]
for feat, val in top10:
    print(f"  {feat:<25s}  {val:.6f}")

summary = pd.DataFrame({
    'Model':    ['Random Forest','SVM (RBF)','Deep MLP'],
    'Features': ['DWT','DWT','DWT'],
    'Accuracy': [rf_acc, svm_acc, mlp_acc],
    'F1':       [rf_f1,  svm_f1,  mlp_f1]
})
summary.to_csv("results/dwt_summary.csv", index=False)
print("\nSaved: results/dwt_summary.csv")
print("\nAll DWT outputs saved to results/")
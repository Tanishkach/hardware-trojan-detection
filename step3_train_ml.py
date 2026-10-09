import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import pickle
import os

os.makedirs("results", exist_ok=True)

with open("results/rf_model.pkl",  "rb") as f: rf     = pickle.load(f)
with open("results/scaler.pkl",    "rb") as f: scaler = pickle.load(f)

df = pd.read_csv("results/features.csv")
feature_cols = [c for c in df.columns if c not in ['chip','subfolder','label']]
X = scaler.transform(df[feature_cols].fillna(0).values)
y = (df['label'] == 'trojan').astype(int).values

rng = np.random.RandomState(42)
idx = rng.choice(len(X), 100, replace=False)
X_s = X[idx]
y_s = y[idx]
print(f"Running SHAP on {len(X_s)} samples...")

explainer = shap.TreeExplainer(
    rf,
    feature_perturbation="tree_path_dependent"
)
shap_values = explainer.shap_values(X_s, check_additivity=False)

# shap_values shape is (100, 30, 2) — take class 1 (Trojan)
print(f"Raw SHAP shape: {np.array(shap_values).shape}")
if isinstance(shap_values, list):
    sv = shap_values[1]           # list of 2 arrays → take index 1
elif np.array(shap_values).ndim == 3:
    sv = shap_values[:, :, 1]     # (N, features, classes) → take class 1
else:
    sv = shap_values
print(f"Using SHAP shape: {sv.shape}")  # should be (100, 30)

ev = explainer.expected_value
if isinstance(ev, (list, np.ndarray)):
    base_val = float(ev[1])
else:
    base_val = float(ev)
print(f"Base value (expected): {base_val:.4f}")

# ---- Plot 1: Bar ----
plt.figure(figsize=(10, 7))
shap.summary_plot(sv, X_s, feature_names=feature_cols,
                  plot_type="bar", show=False, max_display=20)
plt.title("SHAP Feature Importance — Trojan Detection (RF)")
plt.tight_layout()
plt.savefig("results/shap_bar.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/shap_bar.png")

# ---- Plot 2: Beeswarm ----
plt.figure(figsize=(10, 8))
shap.summary_plot(sv, X_s, feature_names=feature_cols,
                  show=False, max_display=20)
plt.title("SHAP Summary — Feature Impact on Trojan Detection")
plt.tight_layout()
plt.savefig("results/shap_summary.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/shap_summary.png")

# ---- Plot 3: Waterfall — one Trojan sample ----
trojan_idx = np.where(y_s == 1)[0]
if len(trojan_idx) > 0:
    i   = trojan_idx[0]
    exp = shap.Explanation(
        values        = sv[i],           # shape (30,)
        base_values   = base_val,
        data          = X_s[i],
        feature_names = feature_cols
    )
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.title("SHAP Waterfall — Single Trojan Sample")
    plt.tight_layout()
    plt.savefig("results/shap_waterfall_trojan.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/shap_waterfall_trojan.png")

# ---- Plot 4: Waterfall — one Clean sample ----
clean_idx = np.where(y_s == 0)[0]
if len(clean_idx) > 0:
    i   = clean_idx[0]
    exp = shap.Explanation(
        values        = sv[i],
        base_values   = base_val,
        data          = X_s[i],
        feature_names = feature_cols
    )
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.title("SHAP Waterfall — Single Clean Sample")
    plt.tight_layout()
    plt.savefig("results/shap_waterfall_clean.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: results/shap_waterfall_clean.png")

# ---- Top 15 features ----
mean_shap = np.abs(sv).mean(axis=0)
top15 = sorted(zip(feature_cols, mean_shap), key=lambda x: -x[1])[:15]
print("\nTop 15 features by SHAP importance:")
print(f"  {'Feature':<30s}  Mean |SHAP|")
for feat, val in top15:
    print(f"  {feat:<30s}  {val:.6f}")

print("\nAll done!")
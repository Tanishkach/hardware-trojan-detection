import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import pickle
import os

os.makedirs("results", exist_ok=True)

with open("results/rf_model.pkl",  "rb") as f: rf     = pickle.load(f)
with open("results/scaler.pkl",    "rb") as f: scaler = pickle.load(f)

df = pd.read_csv("results/features.csv")
feature_cols = [c for c in df.columns if c not in ["chip","subfolder","label"]]
X = scaler.transform(df[feature_cols].fillna(0).values)
y = (df["label"] == "trojan").astype(int).values

rng = np.random.RandomState(42)
idx = rng.choice(len(X), 100, replace=False)
X_s = X[idx]
y_s = y[idx]
print(f"SHAP on {len(X_s)} samples")

explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
shap_raw = explainer.shap_values(X_s, check_additivity=False)

arr = np.array(shap_raw)
print(f"Raw shape: {arr.shape}")

if arr.ndim == 3:
    sv = arr[:, :, 1]
elif isinstance(shap_raw, list):
    sv = np.array(shap_raw[1])
else:
    sv = arr
print(f"sv shape: {sv.shape}")

ev = explainer.expected_value
base_val = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
print(f"base_val: {base_val}")

fig, ax = plt.subplots(figsize=(10, 7))
mean_shap = np.abs(sv).mean(axis=0)
top_idx = np.argsort(mean_shap)[-20:]
top_feats = [feature_cols[i] for i in top_idx]
top_vals  = mean_shap[top_idx]
ax.barh(top_feats, top_vals, color="steelblue")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("SHAP Feature Importance - Trojan Detection (RF)")
plt.tight_layout()
plt.savefig("results/shap_bar.png", dpi=150)
print("Saved shap_bar.png")

fig2, ax2 = plt.subplots(figsize=(12, 8))
pos = np.arange(len(top_feats))
scatter_x = sv[:, top_idx].T
feat_vals  = X_s[:, top_idx].T
for i, (row_x, row_v) in enumerate(zip(scatter_x, feat_vals)):
    colors = ["#d73027" if v > 0 else "#4575b4" for v in row_v]
    ax2.scatter(row_x, [i]*len(row_x), c=colors, alpha=0.6, s=20)
ax2.set_yticks(pos)
ax2.set_yticklabels(top_feats)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("SHAP value (impact on Trojan prediction)")
ax2.set_title("SHAP Summary - Feature Impact on Trojan Detection")
plt.tight_layout()
plt.savefig("results/shap_summary.png", dpi=150)
print("Saved shap_summary.png")

tidx = np.where(y_s == 1)[0][0]
sv1  = sv[tidx]
order = np.argsort(np.abs(sv1))[-15:]
fig3, ax3 = plt.subplots(figsize=(10, 7))
colors3 = ["#d73027" if v > 0 else "#4575b4" for v in sv1[order]]
ax3.barh([feature_cols[i] for i in order], sv1[order], color=colors3)
ax3.axvline(0, color="black", linewidth=0.8)
ax3.set_xlabel("SHAP value")
ax3.set_title("SHAP Waterfall - Single Trojan Sample")
plt.tight_layout()
plt.savefig("results/shap_waterfall_trojan.png", dpi=150)
print("Saved shap_waterfall_trojan.png")

cidx = np.where(y_s == 0)[0][0]
sv0  = sv[cidx]
order0 = np.argsort(np.abs(sv0))[-15:]
fig4, ax4 = plt.subplots(figsize=(10, 7))
colors4 = ["#d73027" if v > 0 else "#4575b4" for v in sv0[order0]]
ax4.barh([feature_cols[i] for i in order0], sv0[order0], color=colors4)
ax4.axvline(0, color="black", linewidth=0.8)
ax4.set_xlabel("SHAP value")
ax4.set_title("SHAP Waterfall - Single Clean Sample")
plt.tight_layout()
plt.savefig("results/shap_waterfall_clean.png", dpi=150)
print("Saved shap_waterfall_clean.png")

mean_shap_all = np.abs(sv).mean(axis=0)
top15 = sorted(zip(feature_cols, mean_shap_all), key=lambda x: -x[1])[:15]
print("\nTop 15 features:")
for feat, val in top15:
    print(f"  {feat:<30s}  {val:.6f}")

print("\nAll done! Check results/ folder.")

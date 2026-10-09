import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

os.makedirs("results", exist_ok=True)

df = pd.read_csv("results/features.csv")
feature_cols = [c for c in df.columns if c not in ['chip','subfolder','label']]

chips = df['chip'].unique()
print(f"Chips found: {list(chips)}\n")

results = []

for test_chip in chips:
    train_df = df[df['chip'] != test_chip]
    test_df  = df[df['chip'] == test_chip]

    if len(test_df) == 0:
        continue

    y_tr = (train_df['label'] == 'trojan').astype(int).values
    y_te = (test_df['label']  == 'trojan').astype(int).values
    X_tr = train_df[feature_cols].fillna(0).values
    X_te = test_df[feature_cols].fillna(0).values

    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average='weighted', zero_division=0)

    print(f"Unseen chip: {test_chip}")
    print(f"  Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
    print(f"  Test size: {len(y_te)}  "
          f"(Clean: {(y_te==0).sum()}, Trojan: {(y_te==1).sum()})")
    print(classification_report(y_te, y_pred,
                                 target_names=['Clean','Trojan'],
                                 zero_division=0))
    results.append({
        'Unseen Chip': test_chip,
        'Accuracy':    round(acc, 4),
        'F1-Score':    round(f1,  4),
        'Test Samples': len(y_te)
    })

res_df = pd.DataFrame(results)
print("="*60)
print("  CROSS-CHIP GENERALIZATION RESULTS")
print("="*60)
print(res_df.to_string(index=False))
res_df.to_csv("results/cross_trojan_results.csv", index=False)

# Bar chart
x     = np.arange(len(res_df))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - width/2, res_df['Accuracy'], width,
            label='Accuracy', color='steelblue', alpha=0.85)
b2 = ax.bar(x + width/2, res_df['F1-Score'],  width,
            label='F1-Score',  color='salmon',    alpha=0.85)

# Value labels on bars
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(res_df['Unseen Chip'], rotation=15, ha='right')
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Cross-Chip Generalization\n"
             "(RF trained on 3 chips, tested on 1 unseen chip)")
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label='Random baseline')
plt.tight_layout()
plt.savefig("results/cross_trojan_results.png", dpi=150)
plt.show()
print("\nSaved: results/cross_trojan_results.csv")
print("Saved: results/cross_trojan_results.png")
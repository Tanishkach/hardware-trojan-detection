import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\bhask\OneDrive\Desktop\HardwareTrojanDetection\data"
os.makedirs("results", exist_ok=True)

def find_trace_files(data_dir):
    files_info = []
    for chip in os.listdir(data_dir):                          # Level 1: AES-T1300_power_Temp25C
        chip_path = os.path.join(data_dir, chip)
        if not os.path.isdir(chip_path): continue

        for sub in os.listdir(chip_path):                     # Level 2: AES-T1300+TrojanDisabled_1
            sub_path = os.path.join(chip_path, sub)
            if not os.path.isdir(sub_path): continue

            if   'TrojanDisabled'  in sub: label = 'clean'
            elif 'TrojanEnabled'   in sub: label = 'trojan'
            elif 'TrojanTriggered' in sub: label = 'trojan'   # triggered = trojan active
            else: continue

            for inner in os.listdir(sub_path):                # Level 3: same name folder OR files
                inner_path = os.path.join(sub_path, inner)

                if os.path.isdir(inner_path):                 # the repeated subfolder
                    for fname in os.listdir(inner_path):
                        if fname.endswith('.csv'):
                            files_info.append({
                                'chip': chip, 'subfolder': sub,
                                'label': label,
                                'filepath': os.path.join(inner_path, fname)
                            })
                elif inner.endswith('.csv'):                   # csv directly at level 3
                    files_info.append({
                        'chip': chip, 'subfolder': sub,
                        'label': label,
                        'filepath': inner_path
                    })
    return files_info

def load_trace(filepath):
    return pd.read_csv(filepath, header=None).values.flatten().astype(np.float64)

files = find_trace_files(DATA_DIR)
print(f"Total trace files found: {len(files)}")

from collections import Counter
print("\nBy label:")
for k, v in Counter(f['label'] for f in files).items():
    print(f"  {k}: {v} files")
print("\nBy chip:")
for k, v in Counter(f['chip'] for f in files).items():
    print(f"  {k}: {v} files")
print("\nBy subfolder type:")
for k, v in Counter(f['subfolder'].split('_')[0].split('+')[1] for f in files).items():
    print(f"  {k}: {v} files")

if not files:
    print("ERROR: Still no files found. Check folder structure.")
    exit()

sample = files[0]
trace  = load_trace(sample['filepath'])
print(f"\nSample: {sample['filepath']}")
print(f"Label : {sample['label']}")
print(f"Length: {len(trace)} samples")
print(f"Values: {trace[:5]}")

# Plot one clean and one trojan
clean_files  = [f for f in files if f['label'] == 'clean']
trojan_files = [f for f in files if f['label'] == 'trojan']

fig, axes = plt.subplots(2, 2, figsize=(14, 6))
for row, (group, name) in enumerate([
        (clean_files,  'Clean (TrojanDisabled)'),
        (trojan_files, 'Trojan (TrojanEnabled/Triggered)')]):
    if not group:
        print(f"Warning: no {name} files found")
        continue
    t = load_trace(group[0]['filepath'])
    axes[row][0].plot(t[:1000], linewidth=0.5)
    axes[row][0].set_title(f"{name} — Time Domain")
    axes[row][0].set_xlabel("Sample"); axes[row][0].set_ylabel("Power")

    fft_vals = np.abs(np.fft.rfft(t))
    axes[row][1].plot(fft_vals[:500], linewidth=0.5, color='orange')
    axes[row][1].set_title(f"{name} — FFT")
    axes[row][1].set_xlabel("Frequency Bin"); axes[row][1].set_ylabel("Magnitude")

plt.tight_layout()
plt.savefig("results/explore_traces.png", dpi=150)
plt.show()
print("\nSaved: results/explore_traces.png")
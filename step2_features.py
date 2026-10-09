import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from tqdm import tqdm

DATA_DIR = r"C:\Users\bhask\OneDrive\Desktop\HardwareTrojanDetection\data"
os.makedirs("results", exist_ok=True)

def find_trace_files(data_dir):
    files_info = []
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
                            files_info.append({
                                'chip': chip, 'subfolder': sub,
                                'label': label,
                                'filepath': os.path.join(inner_path, fname)
                            })
                elif inner.endswith('.csv'):
                    files_info.append({
                        'chip': chip, 'subfolder': sub,
                        'label': label, 'filepath': inner_path
                    })
    return files_info

def load_trace(filepath):
    return pd.read_csv(filepath, header=None).values.flatten().astype(np.float64)

def extract_features(trace):
    trace = trace - np.mean(trace)

    time_feats = {
        'mean':         np.mean(trace),
        'std':          np.std(trace),
        'rms':          np.sqrt(np.mean(trace**2)),
        'skewness':     skew(trace),
        'kurtosis':     kurtosis(trace),
        'peak_to_peak': np.max(trace) - np.min(trace),
        'max':          np.max(trace),
        'min':          np.min(trace),
        'median':       np.median(trace),
        'energy':       np.sum(trace**2),
    }

    fft_vals = np.abs(np.fft.rfft(trace))
    freqs    = np.fft.rfftfreq(len(trace))
    psd      = fft_vals**2
    psd_norm = psd / (np.sum(psd) + 1e-12)
    dom_idx  = np.argmax(fft_vals)

    top_idx  = np.argsort(fft_vals)[-10:][::-1]
    top_mags = {f'fft_top{i+1}_mag': fft_vals[top_idx[i]] for i in range(10)}

    band_size = max(1, len(fft_vals) // 5)
    band_ens  = {f'band{b+1}_energy': np.sum(fft_vals[b*band_size:(b+1)*band_size]**2)
                 for b in range(5)}

    fft_feats = {
        'dominant_freq':      freqs[dom_idx],
        'dominant_magnitude': fft_vals[dom_idx],
        'spectral_entropy':   -np.sum(psd_norm * np.log(psd_norm + 1e-12)),
        'spectral_centroid':  np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-12),
        'total_fft_energy':   np.sum(psd),
        **top_mags,
        **band_ens,
    }
    return {**time_feats, **fft_feats}

# ---- Main ----
files = find_trace_files(DATA_DIR)
print(f"Processing {len(files)} trace files...")

rows = []
for f in tqdm(files):
    try:
        trace = load_trace(f['filepath'])
        if len(trace) < 100: continue
        feats = extract_features(trace)
        rows.append({'chip': f['chip'], 'subfolder': f['subfolder'],
                     'label': f['label'], **feats})
    except Exception as e:
        print(f"  Skipping {f['filepath']}: {e}")

df = pd.DataFrame(rows)
df.to_csv("results/features.csv", index=False)
print(f"\nDone! Shape: {df.shape}")
print("Label distribution:")
print(df['label'].value_counts())
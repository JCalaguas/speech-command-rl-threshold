# -*- coding: utf-8 -*-
"""
Ablation 2 -- Threshold Strategy Comparison
============================================
Evaluate the trained speech-command CNN (best_cnn.keras) on the test set
under three confidence-threshold strategies WITHOUT retraining:

    1. No threshold    (t = 0.00) -- accept every prediction
    2. Fixed threshold (t = 0.50) -- standard baseline
    3. RL-tuned thresh (t = 0.15) -- Q-learning agent result

Metrics per strategy:
    - Accuracy
    - Macro-F1
    - False Reject Rate (FRR)
    - False Accept Rate (FAR)
    - Expected Cost  (FRR * cost_fr + FAR * cost_fa)

Outputs (saved to ./results/):
    - ablation2_bar_chart.png
    - ablation2_threshold_sweep.png
    - ablation2_results.txt
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
except ModuleNotFoundError:
    print("TensorFlow not found. Please install tensorflow first.")
    sys.exit(1)

from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# ==================================================================
# 1.  CONFIGURATION  (must match training notebook)
# ==================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR      = './speech_commands_v0.02'
MODEL_PATH    = './best_cnn.keras'
RESULTS_DIR   = './results'

SAMPLE_RATE   = 16000
CLIP_DURATION = 1.0
N_SAMPLES     = int(SAMPLE_RATE * CLIP_DURATION)

N_MELS        = 64
HOP_LENGTH    = 512
N_FFT         = 1024
FMAX          = 8000

COST_FALSE_REJECT = 2.0
COST_FALSE_ACCEPT = 1.0

COMMANDS = [
    'yes', 'no', 'up', 'down', 'left', 'right',
    'on', 'off', 'stop', 'go',
    'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine',
    'bed', 'bird', 'cat', 'dog', 'happy',
    'house', 'marvin', 'sheila', 'tree', 'wow',
    'backward', 'forward', 'follow', 'learn', 'visual'
]

THRESHOLDS = {
    'No Threshold (t=0.00)':    0.00,
    'Fixed Threshold (t=0.50)': 0.50,
    'RL-Tuned (t=0.15)':        0.15,
}

# ==================================================================
# 2.  HELPERS -- feature extraction (same as training notebook)
# ==================================================================
def load_wav(path, sr=SAMPLE_RATE, n_samples=N_SAMPLES):
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        y = y[:n_samples]
    return y


def wav_to_melspec(y, sr=SAMPLE_RATE, n_mels=N_MELS,
                   n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length, fmax=fmax
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
    return S_db[..., np.newaxis]


def load_file_list(data_dir, commands):
    val_files  = set(open(os.path.join(data_dir, 'validation_list.txt')).read().splitlines())
    test_files = set(open(os.path.join(data_dir, 'testing_list.txt')).read().splitlines())
    records = []
    for cmd in commands:
        cmd_dir = os.path.join(data_dir, cmd)
        if not os.path.isdir(cmd_dir):
            continue
        for fname in os.listdir(cmd_dir):
            if not fname.endswith('.wav'):
                continue
            rel = f'{cmd}/{fname}'
            split = 'val' if rel in val_files else 'test' if rel in test_files else 'train'
            records.append({'path': os.path.join(cmd_dir, fname),
                            'label': cmd, 'split': split})
    return pd.DataFrame(records)


def extract_features(df_split):
    X, y = [], []
    for _, row in tqdm(df_split.iterrows(), total=len(df_split),
                       desc='Extracting test features'):
        wav = load_wav(row['path'])
        spec = wav_to_melspec(wav)
        X.append(spec)
        y.append(row['label_id'])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ==================================================================
# 3.  THRESHOLD-AWARE EVALUATION
# ==================================================================
def evaluate_with_threshold(y_true, y_probs, threshold,
                            cost_fr=COST_FALSE_REJECT,
                            cost_fa=COST_FALSE_ACCEPT):
    """
    Given true labels, softmax probabilities, and a confidence threshold:
      - If max(prob) >= threshold -> predict argmax class
      - Otherwise                -> REJECT (predict -1)

    Returns dict with accuracy, macro_f1, frr, far, expected_cost.
    """
    n = len(y_true)
    max_probs  = y_probs.max(axis=1)
    pred_class = y_probs.argmax(axis=1)

    # Apply threshold: reject when max confidence < threshold
    accepted_mask = max_probs >= threshold
    rejected_mask = ~accepted_mask

    # -- Counts --------------------------------------------------------
    n_accepted = accepted_mask.sum()
    n_rejected = rejected_mask.sum()

    # FALSE REJECT: true command rejected (threshold too aggressive)
    # In this setup every sample IS a valid command (test set has only
    # known classes), so any rejection is a false reject.
    false_rejects = n_rejected
    frr = false_rejects / n if n > 0 else 0.0

    # FALSE ACCEPT: accepted but predicted WRONG class
    if n_accepted > 0:
        wrong_accepted = ((pred_class[accepted_mask] != y_true[accepted_mask]).sum())
        far = wrong_accepted / n if n > 0 else 0.0
    else:
        far = 0.0

    # -- Accuracy & F1 (among accepted samples) ------------------------
    if n_accepted > 0:
        acc = accuracy_score(y_true[accepted_mask], pred_class[accepted_mask])
        # For overall accuracy, count rejects as errors
        overall_acc = (pred_class[accepted_mask] == y_true[accepted_mask]).sum() / n
        macro_f1 = f1_score(y_true[accepted_mask], pred_class[accepted_mask],
                            average='macro', zero_division=0)
        # Adjust macro_f1 to penalise rejections: scale by acceptance rate
        adj_macro_f1 = macro_f1 * (n_accepted / n)
    else:
        overall_acc = 0.0
        adj_macro_f1 = 0.0

    # -- Expected cost -------------------------------------------------
    expected_cost = frr * cost_fr + far * cost_fa

    return {
        'accuracy':      round(float(overall_acc), 4),
        'macro_f1':      round(float(adj_macro_f1), 4),
        'false_reject':  round(float(frr), 4),
        'false_accept':  round(float(far), 4),
        'expected_cost': round(float(expected_cost), 4),
    }


# ==================================================================
# 4.  PLOTTING
# ==================================================================
def plot_bar_chart(results_dict, save_path):
    """Grouped bar chart: one group per metric, one bar per strategy."""
    metrics = ['accuracy', 'macro_f1', 'false_reject', 'false_accept', 'expected_cost']
    labels  = list(results_dict.keys())
    n_groups = len(metrics)
    n_bars   = len(labels)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_groups)
    width = 0.22

    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i, label in enumerate(labels):
        vals = [results_dict[label][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=label, color=colors[i],
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation 2 -- Threshold Strategy Comparison on Test Set',
                 fontsize=14, fontweight='bold')
    nice_labels = ['Accuracy', 'Macro-F1\n(adj.)', 'False Reject\nRate',
                   'False Accept\nRate', 'Expected\nCost']
    ax.set_xticks(x + width)
    ax.set_xticklabels(nice_labels, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, max(max(results_dict[l][m] for l in labels)
                        for m in metrics) * 1.25 + 0.02)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [OK] Bar chart saved -> {save_path}')


def plot_threshold_sweep(y_true, y_probs, save_path,
                         low=0.01, high=0.99, steps=99):
    """Sweep threshold from low to high, plot expected cost & accuracy."""
    thresholds = np.linspace(low, high, steps)
    costs, accs = [], []
    for t in thresholds:
        res = evaluate_with_threshold(y_true, y_probs, threshold=t)
        costs.append(res['expected_cost'])
        accs.append(res['accuracy'])

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color_cost = '#E53935'
    color_acc  = '#1E88E5'

    ax1.set_xlabel('Confidence Threshold (t)', fontsize=12)
    ax1.set_ylabel('Expected Cost', color=color_cost, fontsize=12)
    ax1.plot(thresholds, costs, color=color_cost, linewidth=2, label='Expected Cost')
    ax1.tick_params(axis='y', labelcolor=color_cost)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color=color_acc, fontsize=12)
    ax2.plot(thresholds, accs, color=color_acc, linewidth=2, linestyle='--',
             label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    # Mark the three strategies
    markers = {0.00: ('No Threshold', 'v', '#9C27B0'),
               0.15: ('RL-Tuned t=0.15', 'o', '#FF9800'),
               0.50: ('Fixed t=0.50', 's', '#4CAF50')}
    for tau, (name, marker, col) in markers.items():
        res = evaluate_with_threshold(y_true, y_probs, threshold=tau)
        ax1.scatter(tau, res['expected_cost'], marker=marker, s=120,
                    color=col, zorder=5, edgecolors='black', linewidth=1.2)
        ax2.scatter(tau, res['accuracy'], marker=marker, s=120,
                    color=col, zorder=5, edgecolors='black', linewidth=1.2)
        ax1.annotate(name, (tau, res['expected_cost']),
                     textcoords='offset points', xytext=(10, 12),
                     fontsize=9, fontweight='bold', color=col)

    ax1.set_title('Ablation 2 -- Threshold Sweep: Expected Cost & Accuracy',
                  fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc='center right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  [OK] Sweep plot saved -> {save_path}')


# ==================================================================
# 5.  MAIN
# ==================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -- Load model ----------------------------------------------------
    print('\n[1/4] Loading trained model ...')
    model = keras.models.load_model(MODEL_PATH)
    print(f'       Model loaded from {MODEL_PATH}')

    # -- Prepare test data ---------------------------------------------
    print('\n[2/4] Preparing test features ...')
    cache_x = os.path.join(RESULTS_DIR, '_X_test_cache.npy')
    cache_y = os.path.join(RESULTS_DIR, '_y_test_cache.npy')

    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print('       Loading cached test features ...')
        X_test = np.load(cache_x)
        y_test = np.load(cache_y)
    else:
        print('       Extracting test features from audio (this takes ~1-2 min) ...')
        df = load_file_list(DATA_DIR, COMMANDS)
        le = LabelEncoder()
        df['label_id'] = le.fit_transform(df['label'])
        X_test, y_test = extract_features(df[df['split'] == 'test'])
        np.save(cache_x, X_test)
        np.save(cache_y, y_test)
        print(f'       Cached to {cache_x} & {cache_y}')

    print(f'       X_test shape: {X_test.shape}  |  y_test shape: {y_test.shape}')

    # -- Get predictions -----------------------------------------------
    print('\n[3/4] Running inference on test set ...')
    y_probs = model.predict(X_test, batch_size=256, verbose=1)
    print(f'       Predictions shape: {y_probs.shape}')

    # -- Evaluate threshold strategies ---------------------------------
    print('\n[4/4] Evaluating threshold strategies ...\n')
    results = {}
    for name, tau in THRESHOLDS.items():
        res = evaluate_with_threshold(y_test, y_probs, threshold=tau)
        results[name] = res

    # -- Print results table -------------------------------------------
    header = f'{"Strategy":<30} {"Accuracy":>10} {"Macro-F1":>10} {"FRR":>10} {"FAR":>10} {"Exp.Cost":>10}'
    sep    = '-' * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, res in results.items():
        print(f'{name:<30} {res["accuracy"]:>10.4f} {res["macro_f1"]:>10.4f} '
              f'{res["false_reject"]:>10.4f} {res["false_accept"]:>10.4f} '
              f'{res["expected_cost"]:>10.4f}')
    print(sep)

    # Save text report
    report_path = os.path.join(RESULTS_DIR, 'ablation2_results.txt')
    with open(report_path, 'w') as f:
        f.write('Ablation 2 -- Threshold Strategy Comparison\n')
        f.write(f'Model: {MODEL_PATH}\n')
        f.write(f'Test samples: {len(y_test)}\n')
        f.write(f'Cost(false_reject)={COST_FALSE_REJECT}, Cost(false_accept)={COST_FALSE_ACCEPT}\n\n')
        f.write(sep + '\n')
        f.write(header + '\n')
        f.write(sep + '\n')
        for name, res in results.items():
            f.write(f'{name:<30} {res["accuracy"]:>10.4f} {res["macro_f1"]:>10.4f} '
                    f'{res["false_reject"]:>10.4f} {res["false_accept"]:>10.4f} '
                    f'{res["expected_cost"]:>10.4f}\n')
        f.write(sep + '\n')
    print(f'\n  [OK] Text report saved -> {report_path}')

    # -- Plots ---------------------------------------------------------
    print('\n  Generating plots ...')
    plot_bar_chart(results, os.path.join(RESULTS_DIR, 'ablation2_bar_chart.png'))
    plot_threshold_sweep(y_test, y_probs,
                         os.path.join(RESULTS_DIR, 'ablation2_threshold_sweep.png'))

    print('\n[DONE] Ablation 2 complete!\n')


if __name__ == '__main__':
    main()

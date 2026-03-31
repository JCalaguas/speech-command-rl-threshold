# -*- coding: utf-8 -*-
"""
Error & Slice Analysis -- Speech Command Recognition
=====================================================
Based on error_slice_analysis.ipynb, converted to standalone script.

Covers:
- Confusion matrix heatmap
- Worst performing commands (bottom 5)
- Best performing commands (top 5)
- Per-speaker failure cases
- Most confused command pairs
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import librosa
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
except ModuleNotFoundError:
    print("TensorFlow not found. Please install tensorflow first.")
    sys.exit(1)

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

# ==================================================================
# CONFIG -- must match training
# ==================================================================
DATA_DIR      = './speech_commands_v0.02'
MODEL_PATH    = './best_cnn.keras'
RESULTS_DIR   = './results'
SAMPLE_RATE   = 16000
N_SAMPLES     = 16000
N_MELS        = 64
HOP_LENGTH    = 512
N_FFT         = 1024
FMAX          = 8000
BATCH_SIZE    = 64
RL_THRESHOLD  = 0.15
COST_FR       = 2.0
COST_FA       = 1.0
SEED          = 42

COMMANDS = [
    'yes', 'no', 'up', 'down', 'left', 'right',
    'on', 'off', 'stop', 'go',
    'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine',
    'bed', 'bird', 'cat', 'dog', 'happy',
    'house', 'marvin', 'sheila', 'tree', 'wow',
    'backward', 'forward', 'follow', 'learn', 'visual'
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================================================================
# HELPERS
# ==================================================================
def load_wav(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    return y[:N_SAMPLES]

def wav_to_melspec(y):
    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
                                       n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX)
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

# ==================================================================
# MAIN
# ==================================================================
def main():
    # -- Load model ------------------------------------------------
    print('\n[1/6] Loading model ...')
    model = keras.models.load_model(MODEL_PATH)
    print(f'      Model loaded from {MODEL_PATH}')

    # -- Prepare test data -----------------------------------------
    print('\n[2/6] Preparing test data ...')
    cache_x = os.path.join(RESULTS_DIR, '_X_test_cache.npy')
    cache_y = os.path.join(RESULTS_DIR, '_y_test_cache.npy')

    df_all = load_file_list(DATA_DIR, COMMANDS)
    le = LabelEncoder()
    df_all['label_id'] = le.fit_transform(df_all['label'])
    df_test_raw = df_all[df_all['split'] == 'test'].copy().reset_index(drop=True)

    if os.path.exists(cache_x) and os.path.exists(cache_y):
        print('      Loading cached test features ...')
        X_test = np.load(cache_x)
        y_test = np.load(cache_y)
    else:
        print('      Extracting test features (this takes ~1-2 min) ...')
        X_list, y_list = [], []
        for _, row in tqdm(df_test_raw.iterrows(), total=len(df_test_raw),
                           desc='Extracting test features'):
            X_list.append(wav_to_melspec(load_wav(row['path'])))
            y_list.append(row['label_id'])
        X_test = np.array(X_list, dtype=np.float32)
        y_test = np.array(y_list, dtype=np.int32)
        np.save(cache_x, X_test)
        np.save(cache_y, y_test)

    print(f'      X_test: {X_test.shape}  |  y_test: {y_test.shape}')

    # -- Inference -------------------------------------------------
    print('\n[3/6] Running inference ...')
    probs_test = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    max_probs  = probs_test.max(axis=1)
    pred_class = probs_test.argmax(axis=1)
    accepted   = max_probs >= RL_THRESHOLD

    y_true_acc = y_test[accepted]
    y_pred_acc = pred_class[accepted]

    # ==========================================================
    # 1. CONFUSION MATRIX
    # ==========================================================
    print('\n[4/6] Generating confusion matrix ...')
    cm = confusion_matrix(y_true_acc, y_pred_acc)

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix -- RL-Tuned Threshold (t={RL_THRESHOLD})\n'
                 f'({accepted.sum():,} accepted / {len(y_test):,} total)',
                 fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('      Saved -> ./results/confusion_matrix.png')

    # ==========================================================
    # 2. PER-CLASS PERFORMANCE -- Best & Worst
    # ==========================================================
    print('\n[5/6] Per-class performance analysis ...')
    f1_per_class = f1_score(y_true_acc, y_pred_acc, average=None,
                            labels=list(range(len(le.classes_))),
                            zero_division=0)

    per_class_df = pd.DataFrame({
        'command': le.classes_,
        'f1_score': f1_per_class
    }).sort_values('f1_score', ascending=False).reset_index(drop=True)

    per_class_acc = []
    for i, cmd in enumerate(le.classes_):
        mask = y_true_acc == i
        if mask.sum() > 0:
            acc = (y_pred_acc[mask] == i).mean()
        else:
            acc = 0.0
        per_class_acc.append(acc)

    per_class_df['accuracy'] = [per_class_acc[list(le.classes_).index(c)]
                                for c in per_class_df['command']]

    top5    = per_class_df.head(5)
    bottom5 = per_class_df.tail(5).sort_values('f1_score')

    print('\n      -- Top 5 Best Commands --')
    print(top5[['command', 'f1_score', 'accuracy']].to_string(index=False,
          float_format=lambda x: f'{x:.4f}'))
    print('\n      -- Bottom 5 Worst Commands --')
    print(bottom5[['command', 'f1_score', 'accuracy']].to_string(index=False,
          float_format=lambda x: f'{x:.4f}'))

    # Plot best vs worst
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    bars1 = ax1.barh(top5['command'], top5['f1_score'],
                      color='steelblue', edgecolor='white')
    ax1.set_xlim(0, 1.05)
    ax1.set_title('Top 5 Best Commands -- F1 Score', fontweight='bold')
    ax1.set_xlabel('F1 Score')
    ax1.bar_label(bars1, fmt='%.3f', padding=3)
    ax1.axvline(per_class_df['f1_score'].mean(), color='red',
                linestyle='--', label=f'Mean = {per_class_df["f1_score"].mean():.3f}')
    ax1.legend()

    bars2 = ax2.barh(bottom5['command'], bottom5['f1_score'],
                      color='crimson', edgecolor='white')
    ax2.set_xlim(0, 1.05)
    ax2.set_title('Bottom 5 Worst Commands -- F1 Score', fontweight='bold')
    ax2.set_xlabel('F1 Score')
    ax2.bar_label(bars2, fmt='%.3f', padding=3)
    ax2.axvline(per_class_df['f1_score'].mean(), color='red',
                linestyle='--', label=f'Mean = {per_class_df["f1_score"].mean():.3f}')
    ax2.legend()

    plt.suptitle('Per-Command F1 Score -- Best vs Worst', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'per_class_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('      Saved -> ./results/per_class_performance.png')

    # Full per-class bar chart
    fig, ax = plt.subplots(figsize=(16, 7))
    colors = ['steelblue' if f >= per_class_df['f1_score'].mean() else 'crimson'
              for f in per_class_df['f1_score']]
    bars = ax.bar(per_class_df['command'], per_class_df['f1_score'],
                  color=colors, edgecolor='white', alpha=0.85)
    ax.axhline(per_class_df['f1_score'].mean(), color='black',
               linestyle='--', lw=2, label=f'Mean F1 = {per_class_df["f1_score"].mean():.3f}')
    ax.set_title('Per-Command F1 Score -- All 35 Commands', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', rotation=45)

    blue_patch = mpatches.Patch(color='steelblue', label='Above mean')
    red_patch  = mpatches.Patch(color='crimson',   label='Below mean')
    ax.legend(handles=[blue_patch, red_patch,
                       plt.Line2D([0], [0], color='black', linestyle='--', lw=2,
                                  label=f'Mean = {per_class_df["f1_score"].mean():.3f}')])
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'all_class_f1.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('      Saved -> ./results/all_class_f1.png')

    # ==========================================================
    # 3. MOST CONFUSED COMMAND PAIRS
    # ==========================================================
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    confused_pairs = []
    for true_label in le.classes_:
        for pred_label in le.classes_:
            if true_label != pred_label:
                count = cm_df.loc[true_label, pred_label]
                if count > 0:
                    confused_pairs.append({
                        'true': true_label,
                        'pred': pred_label,
                        'count': count
                    })

    confused_df = pd.DataFrame(confused_pairs).sort_values('count', ascending=False).head(10)

    print('\n      -- Top 10 Most Confused Pairs --')
    print(f'      {"True Label":<15} {"Predicted As":<15} {"Count":>8}')
    print('      ' + '-' * 42)
    for _, row in confused_df.iterrows():
        print(f'      {row["true"]:<15} {row["pred"]:<15} {int(row["count"]):>8}')

    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [f'{r["true"]} -> {r["pred"]}' for _, r in confused_df.iterrows()]
    bars = ax.barh(labels, confused_df['count'], color='darkorange', edgecolor='white')
    ax.set_title('Top 10 Most Confused Command Pairs', fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of Misclassifications')
    ax.bar_label(bars, padding=3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confused_pairs.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('      Saved -> ./results/confused_pairs.png')

    # ==========================================================
    # 4. PER-SPEAKER FAILURE ANALYSIS
    # ==========================================================
    print('\n[6/6] Per-speaker slice analysis ...')
    df_test_raw = df_test_raw.iloc[:len(probs_test)].copy()
    df_test_raw['speaker']      = df_test_raw['path'].apply(lambda p: os.path.basename(p).split('_')[0])
    df_test_raw['pred_class']   = pred_class
    df_test_raw['confidence']   = max_probs
    df_test_raw['accepted']     = accepted
    df_test_raw['correct']      = pred_class == y_test
    df_test_raw['false_reject'] = ~accepted
    df_test_raw['false_accept'] = (pred_class != y_test) & accepted

    speaker_stats = df_test_raw.groupby('speaker').agg(
        n_samples    = ('label_id', 'count'),
        accuracy     = ('correct',   'mean'),
        fr_rate      = ('false_reject', 'mean'),
        fa_rate      = ('false_accept', 'mean')
    ).query('n_samples >= 10').reset_index()

    print(f'      Speakers with >=10 test samples: {len(speaker_stats)}')
    print(f'      Mean accuracy across speakers:   {speaker_stats["accuracy"].mean():.4f}')
    print(f'      Std accuracy across speakers:    {speaker_stats["accuracy"].std():.4f}')
    print(f'      Mean FR rate across speakers:    {speaker_stats["fr_rate"].mean():.4f}')
    print()
    print('      -- 10 Worst Speakers by Accuracy --')
    print(speaker_stats.nsmallest(10, 'accuracy')
          [['speaker', 'n_samples', 'accuracy', 'fr_rate']]
          .to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(speaker_stats['accuracy'], bins=25, color='steelblue', edgecolor='white')
    axes[0].axvline(speaker_stats['accuracy'].mean(), color='red', linestyle='--', lw=2,
                    label=f'Mean = {speaker_stats["accuracy"].mean():.3f}')
    axes[0].set_title('Per-Speaker Accuracy Distribution', fontweight='bold')
    axes[0].set_xlabel('Accuracy')
    axes[0].legend()

    axes[1].hist(speaker_stats['fr_rate'], bins=25, color='darkorange', edgecolor='white')
    axes[1].axvline(speaker_stats['fr_rate'].mean(), color='red', linestyle='--', lw=2,
                    label=f'Mean = {speaker_stats["fr_rate"].mean():.3f}')
    axes[1].set_title('Per-Speaker False Reject Rate', fontweight='bold')
    axes[1].set_xlabel('False Reject Rate')
    axes[1].legend()

    axes[2].scatter(speaker_stats['n_samples'], speaker_stats['accuracy'],
                    alpha=0.5, color='purple', edgecolors='white', s=40)
    axes[2].set_title('Accuracy vs Number of Test Samples', fontweight='bold')
    axes[2].set_xlabel('Number of Test Samples')
    axes[2].set_ylabel('Accuracy')

    plt.suptitle('Per-Speaker Slice Analysis -- Fairness Audit', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'speaker_slice_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('      Saved -> ./results/speaker_slice_analysis.png')

    # ==========================================================
    # 5. SUMMARY
    # ==========================================================
    print('\n' + '=' * 55)
    print('  ERROR / SLICE ANALYSIS SUMMARY')
    print('=' * 55)
    print(f'  Test samples evaluated : {len(y_test):,}')
    print(f'  Accepted (thr={RL_THRESHOLD})     : {accepted.sum():,} ({accepted.mean()*100:.1f}%)')
    print(f'  Rejected               : {(~accepted).sum():,} ({(~accepted).mean()*100:.1f}%)')
    print()
    print('  Per-Class Performance')
    print(f'    Mean F1  : {per_class_df["f1_score"].mean():.4f}')
    print(f'    Best cmd : {per_class_df.iloc[0]["command"]} (F1={per_class_df.iloc[0]["f1_score"]:.4f})')
    print(f'    Worst cmd: {per_class_df.iloc[-1]["command"]} (F1={per_class_df.iloc[-1]["f1_score"]:.4f})')
    print()
    print('  Most Confused Pair')
    top_pair = confused_df.iloc[0]
    print(f'    "{top_pair["true"]}" misclassified as "{top_pair["pred"]}" {int(top_pair["count"])} times')
    print()
    print('  Speaker Fairness')
    print(f'    Speakers analyzed : {len(speaker_stats)}')
    print(f'    Accuracy std      : {speaker_stats["accuracy"].std():.4f}')
    print(f'    FR rate std       : {speaker_stats["fr_rate"].std():.4f}')
    print()
    print('  Plots saved to ./results/')
    print('    confusion_matrix.png')
    print('    per_class_performance.png')
    print('    all_class_f1.png')
    print('    confused_pairs.png')
    print('    speaker_slice_analysis.png')
    print('=' * 55)
    print('\n[DONE] Error Slice Analysis complete!\n')


if __name__ == '__main__':
    main()

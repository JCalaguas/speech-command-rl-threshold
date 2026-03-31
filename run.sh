#!/bin/bash
# ============================================================
# run.sh — One-command reproduction script
# Speech Command Recognition + RL Threshold Tuning
# ============================================================
# Usage:
#   bash run.sh          → full pipeline (download + train + eval)
#   bash run.sh --skip-download  → skip dataset download
#   bash run.sh --eval-only      → skip training, load best_cnn.keras
# ============================================================

set -e  # exit on error

SKIP_DOWNLOAD=false
EVAL_ONLY=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --eval-only)     EVAL_ONLY=true ;;
    esac
done

echo "============================================"
echo "  Speech Command Recognition — Full Pipeline"
echo "============================================"

# ── Step 1: Install dependencies ─────────────────────────────
echo ""
echo "[1/5] Installing dependencies..."
pip install -r requirements.txt -q
echo "      Done."

# ── Step 2: Download dataset ─────────────────────────────────
if [ "$SKIP_DOWNLOAD" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "[2/5] Downloading Speech Commands V2 dataset..."
    python - <<'PYEOF'
import os
import urllib.request
import tarfile

url           = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
archive       = 'speech_commands_v0.02.tar.gz'
extracted_dir = 'speech_commands_v0.02'

if not os.path.exists(archive):
    print(f'  Downloading {url}...')
    urllib.request.urlretrieve(url, archive)
    print('  Download complete.')
else:
    print('  Archive already downloaded — skipping.')

if not os.path.exists(os.path.join(extracted_dir, 'validation_list.txt')):
    print('  Extracting...')
    os.makedirs(extracted_dir, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(path=extracted_dir)
    print('  Extraction complete.')
else:
    print('  Dataset already extracted — skipping.')
PYEOF
    echo "      Done."
else
    echo "[2/5] Skipping dataset download."
fi

# ── Step 3: Run training ──────────────────────────────────────
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "[3/5] Running training pipeline..."
    echo "      (Feature extraction + CNN training + RL agent)"
    echo "      Expected time: ~60-90 minutes on CPU, ~20-30 min on GPU"
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=7200 \
        --output speech_command_cnn_rl_executed.ipynb \
        speech_command_cnn_rl_vscode.ipynb
    echo "      Training complete. Model saved to best_cnn.keras"
else
    echo "[3/5] Skipping training (--eval-only). Loading best_cnn.keras..."
    if [ ! -f "best_cnn.keras" ]; then
        echo "      ERROR: best_cnn.keras not found. Run without --eval-only first."
        exit 1
    fi
fi

# ── Step 4: Run ablation study ────────────────────────────────
echo ""
echo "[4/5] Running ablation study..."
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    --output ablation_study_executed.ipynb \
    ablation_study.ipynb
echo "      Ablation plots saved to ./results/"

# ── Step 5: Run error/slice analysis ─────────────────────────
echo ""
echo "[5/5] Running error/slice analysis..."
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=1800 \
    --output error_slice_analysis_executed.ipynb \
    error_slice_analysis.ipynb
echo "      Error analysis plots saved to ./results/"

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results saved to: ./results/"
echo "  Trained model:    ./best_cnn.keras"
echo "  Executed notebooks:"
echo "    speech_command_cnn_rl_executed.ipynb"
echo "    ablation_study_executed.ipynb"
echo "    error_slice_analysis_executed.ipynb"
echo "============================================"

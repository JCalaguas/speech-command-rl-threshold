# ============================================================
# Makefile — Speech Command Recognition + RL Threshold Tuning
# ============================================================
# Commands:
#   make install     → install all dependencies
#   make data        → download and extract dataset
#   make train       → run full training pipeline
#   make ablation    → run ablation study
#   make eval        → run error/slice analysis
#   make repro       → full reproduction (install+data+train+ablation+eval)
#   make demo        → run terminal demo (mic mode)
#   make clean       → remove generated files
# ============================================================

.PHONY: install data train ablation eval repro demo clean

# ── Install dependencies ──────────────────────────────────────
install:
	@echo "[install] Installing dependencies..."
	pip install -r requirements.txt -q
	@echo "[install] Done."

# ── Download dataset ──────────────────────────────────────────
data:
	@echo "[data] Downloading Speech Commands V2..."
	python -c "\
import os, urllib.request, tarfile; \
url='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'; \
archive='speech_commands_v0.02.tar.gz'; \
d='speech_commands_v0.02'; \
(urllib.request.urlretrieve(url, archive) if not os.path.exists(archive) else print('Already downloaded.')); \
(([os.makedirs(d, exist_ok=True), tarfile.open(archive,'r:gz').extractall(path=d)]) if not os.path.exists(os.path.join(d,'validation_list.txt')) else print('Already extracted.'))"
	@echo "[data] Done."

# ── Train ─────────────────────────────────────────────────────
train:
	@echo "[train] Running training pipeline (~60-90 min)..."
	jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=7200 \
		--output speech_command_cnn_rl_executed.ipynb \
		speech_command_cnn_rl_vscode.ipynb
	@echo "[train] Done. Model saved to best_cnn.keras"

# ── Ablation study ────────────────────────────────────────────
ablation:
	@echo "[ablation] Running ablation study..."
	jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=3600 \
		--output ablation_study_executed.ipynb \
		ablation_study.ipynb
	@echo "[ablation] Plots saved to ./results/"

# ── Error/slice analysis ──────────────────────────────────────
eval:
	@echo "[eval] Running error/slice analysis..."
	jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=1800 \
		--output error_slice_analysis_executed.ipynb \
		error_slice_analysis.ipynb
	@echo "[eval] Plots saved to ./results/"

# ── Full reproduction ─────────────────────────────────────────
repro: install data train ablation eval
	@echo ""
	@echo "============================================"
	@echo "  Full reproduction complete!"
	@echo "  Model:   ./best_cnn.keras"
	@echo "  Results: ./results/"
	@echo "============================================"

# ── Demo ──────────────────────────────────────────────────────
demo:
	@echo "[demo] Starting terminal demo (microphone mode)..."
	python demo.py --mic

# ── Clean generated files ─────────────────────────────────────
clean:
	@echo "[clean] Removing generated files..."
	rm -f best_cnn.keras
	rm -f speech_command_cnn_rl_executed.ipynb
	rm -f ablation_study_executed.ipynb
	rm -f error_slice_analysis_executed.ipynb
	rm -f live_mic.wav
	rm -rf results/
	@echo "[clean] Done."

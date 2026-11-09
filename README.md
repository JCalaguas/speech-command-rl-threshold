# Speech Command Recognition with RL-Optimized Threshold Tuning

A speech command recognition pipeline where a **Q-learning agent** learns the best
confidence threshold for accepting or rejecting a CNN's prediction, then checks the
result for fairness across speakers.

## How it works
1. **Dataset:** Google Speech Commands v0.02
2. **Features:** WAV audio → Mel spectrograms (librosa)
3. **Classifier:** CNN built in TensorFlow/Keras
4. **Baseline:** evaluation at the default 0.5 confidence threshold
5. **RL agent:** Q-learning tunes the decision threshold to balance false accepts
   against false rejects
6. **Final evaluation:** baseline vs RL-tuned threshold
7. **Ethics audit:** per-speaker fairness check

## Files
| File | Use |
|---|---|
| `speech_command_cnn_rl.ipynb` | Run on Kaggle (uses the Speech Commands v2 Kaggle dataset) |
| `speech_command_cnn_rl_vscode.ipynb` | Run locally in VS Code / Jupyter |

## Setup
1. Download the [Speech Commands v0.02 dataset](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
2. Install: `pip install tensorflow librosa numpy pandas scikit-learn matplotlib seaborn tqdm`
3. In the VS Code notebook, set `DATA_DIR` to where you extracted the dataset, then run all cells

## Built with
Python · TensorFlow/Keras · librosa · Q-learning

## Team
Group project for **Intelligent Systems**, Holy Angel University (2026).
My role: speech recognition module, the audio preprocessing pipeline, and
connecting the CNN to the RL agent.

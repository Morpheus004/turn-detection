# 🎙️ Turn Detection Model - Shiprocket Challenge

A tiny, fast, and accurate audio-based Turn Detection model that determines whether a speaker has completed their turn or is simply pausing (e.g. thinking, using filler words like "um", "uh", "matlab", "toh", etc.).

## 🏗️ Architecture Overview

The system processes raw 16kHz mono audio (up to 8 seconds context window) directly in the waveform/mel domain to capture acoustic, semantic, and prosodic cues:

```
Raw Audio (16kHz, ≤8s)
   │
   ▼
Whisper Mel-Spectrogram (80 mels × 800 frames)
   │
   ▼
Whisper-Tiny Audio Encoder (or Custom CNN+GRU)
   │
   ▼
Learned Attention Pooling (Temporal weighting of cues)
   │
   ▼
MLP Classification Head (Linear → LayerNorm → GELU → Dropout → Linear)
   │
   ▼
Turn Completion Probability [0.0 - 1.0] (Sigmoid)
```

## 📁 Repository Structure

```
shiprocket/
├── src/
│   ├── __init__.py
│   ├── model.py          # SmartTurnModel & LightweightTurnModel architectures
│   ├── data.py           # HuggingFace dataset loading, 8s padding, feature extraction
│   ├── train.py          # Training loop, cosine LR scheduler, checkpointing, W&B
│   ├── evaluate.py       # Precision, Recall, F1, latency benchmarks, plotting
│   ├── inference.py      # Real-time TurnDetector class for single/batch audio
│   └── utils.py          # Seeds, device configuration, parameter counters
├── scripts/
│   └── run_experiment.py # Unified CLI experiment runner
├── checkpoints/          # Saved model weights (.pt / .onnx)
├── EXPERIMENTS.md        # Detailed hypothesis registry & experiment graph
├── RESULTS.md            # Benchmark tables and metric logs
├── README.md             # Project documentation
└── pyproject.toml        # uv / pip environment dependencies
```

## 🚀 Getting Started

### 1. Installation

Using `uv`:
```bash
uv sync
```

Or using standard `pip`:
```bash
pip install -r pyproject.toml
```

### 2. Running Experiments

**Baseline (Frozen Whisper Tiny + Attention Head)**:
```bash
uv run python scripts/run_experiment.py --experiment exp001_baseline --epochs 4 --batch-size 32
```

**Fine-Tuning (Unfreeze Last 2 Encoder Layers)**:
```bash
uv run python scripts/run_experiment.py --experiment exp002b_unfreeze2 --epochs 6 --no-freeze --unfreeze-last-n 2 --lr 2e-5
```

**Lightweight Custom Architecture (From Scratch)**:
```bash
uv run python scripts/run_experiment.py --experiment exp004_lightweight --model-type lightweight --epochs 20 --lr 1e-3 --batch-size 64
```

### 3. Inference Example

```python
import numpy as np
from src.inference import TurnDetector

# Initialize detector
detector = TurnDetector("checkpoints/exp001_baseline_final.pt")

# 16kHz audio sample (numpy array)
audio = np.zeros(16000 * 3, dtype=np.float32)

# Predict turn completion
result = detector.predict(audio)
print(result)
# {'prediction': 1, 'probability': 0.88, 'inference_time_ms': 12.4}
```

## 📊 Dataset Reference

- **Source**: [`pipecat-ai/smart-turn-data-v3.2-train`](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
- **Target label**: `endpoint_bool` (1: Complete / Turn End, 0: Incomplete / Pause)
- **Special features**: Includes annotations for `midfiller`, `endfiller`, `language`, and `synthetic` samples.

## 🛠️ Code Quality, Formatting & Linting

This project uses **[Ruff](https://astral.sh/ruff)** for blazing-fast linting and code formatting, **[Pyright](https://github.com/microsoft/pyright)** for static type checks (native engine behind VS Code / Cursor Pylance), and **[pre-commit](https://pre-commit.com/)** for automated Git hooks.

### Commands

Install dev dependencies & Git hooks:
```bash
make install-dev
# or: uv sync --group dev && uv run pre-commit install
```

Auto-format all code (isort + black-compatible styling):
```bash
make format
# or: uv run ruff format . && uv run ruff check --fix .
```

Run linter & static type checks:
```bash
make lint
# or: uv run ruff check . && uv run pyright src scripts app.py
```

Run all formatting and lint checks in one go:
```bash
make check
```

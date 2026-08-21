# Experiment Results

This document tracks all experiment results for the Shiprocket Turn Detection challenge.

## Summary Table

| Experiment | Model Description | Parameters (Total / Trainable) | Accuracy | Precision | Recall | F1 Score | Latency GPU (ms) | Status |
|---|---|---|---|---|---|---|---|---|
| EXP-001 | Whisper Tiny (frozen) + Head | ~37.8M / ~100K | - | - | - | - | - | 🔲 Pending |
| EXP-002a | Whisper Tiny (unfreeze 1 layer) + Head | ~37.8M / ~3.2M | - | - | - | - | - | 🔲 Pending |
| EXP-002b | Whisper Tiny (unfreeze 2 layers) + Head | ~37.8M / ~6.4M | - | - | - | - | - | 🔲 Pending |
| EXP-003 | Whisper Tiny (full fine-tune) | ~37.8M / ~37.8M | - | - | - | - | - | 🔲 Pending |
| EXP-004 | Custom CNN+GRU (from scratch) | ~500K / ~500K | - | - | - | - | - | 🔲 Pending |
| EXP-005 | Best Architecture + Hindi/Hinglish focus | - | - | - | - | - | - | 🔲 Pending |
| EXP-006 | Best Architecture + Augmentation | - | - | - | - | - | - | 🔲 Pending |
| EXP-007 | Optimal Threshold Tuned | - | - | - | - | - | - | 🔲 Pending |
| EXP-008 | ONNX INT8 Quantized | ~8MB | - | - | - | - | - | 🔲 Pending |

---

## Detailed Experiment Logs

### EXP-001: Baseline - Whisper Tiny (Frozen) + Classification Head
- **Date**: -
- **Status**: Pending
- **Configuration**:
  - Backbone: `openai/whisper-tiny` (Encoder frozen)
  - Head: Attention pooling + MLP (Linear 384→256→64→1)
  - Optimizer: AdamW (lr=5e-5, weight_decay=0.01)
  - Batch Size: 32
  - Epochs: 4
  - Warmup: 20% cosine decay
- **Validation Metrics**:
  - Accuracy: -
  - Precision: -
  - Recall: -
  - F1 Score: -
- **Inference Speed**: - ms / utterance
- **Observations & Analysis**: -

---

*New experiment outputs will be logged automatically here upon completion.*

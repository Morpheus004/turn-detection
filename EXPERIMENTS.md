# Experiments Log

This document tracks all experiments for the Shiprocket Turn Detection challenge.

## Problem Statement
Build a tiny, audio-based Turn Detection model from scratch. Turn detection decides whether the user is actually done speaking or just pausing. Special focus on Indian Hinglish, filler words, and pauses.

## Dataset
- **Source**: `pipecat-ai/smart-turn-data-v3.2-train` & `pipecat-ai/smart-turn-data-v3.2-test` (HuggingFace)
- **Columns**: audio, id, language, endpoint_bool (label), midfiller, endfiller, synthetic, dataset, spoken_text
- **Task**: Binary classification - predict endpoint_bool (True = user done speaking, False = still speaking/pausing)
- **Audio format**: 16kHz mono, up to 8 seconds

## Experiment Registry

### EXP-001: Baseline - Whisper Tiny Encoder (Frozen) + Classification Head
- **Status**: 🔲 Not Started
- **Hypothesis**: Using frozen Whisper Tiny encoder features with an attention-pooling + classifier head should give a reasonable baseline (~85% accuracy) since Whisper already understands speech patterns.
- **Architecture**: Whisper Tiny Encoder (frozen, ~37M params) → Attention Pooling → Classifier Head (Linear 384→256→64→1)
- **Config**: lr=5e-5, epochs=4, batch_size=32, warmup_ratio=0.2
- **What we learn**: How good are pre-trained Whisper features for turn detection out of the box.

### EXP-002: Whisper Tiny Encoder (Partial Fine-tune) + Classification Head
- **Status**: 🔲 Not Started
- **Hypothesis**: Unfreezing last 1-2 encoder layers should improve performance as the model adapts its acoustic/linguistic representations for turn detection specifically.
- **Variants**:
  - EXP-002a: Unfreeze last 1 layer
  - EXP-002b: Unfreeze last 2 layers
  - EXP-002c: Unfreeze last 4 layers (all)
- **Config**: lr=2e-5 (lower for fine-tuning), epochs=6, batch_size=32
- **What we learn**: Sweet spot for how much of the encoder to fine-tune without overfitting.

### EXP-003: Whisper Tiny Full Fine-tune
- **Status**: 🔲 Not Started
- **Hypothesis**: Fully fine-tuning the entire encoder might give best accuracy but requires careful regularization.
- **Config**: lr=1e-5 (very low), epochs=4, batch_size=16
- **What we learn**: Upper bound of Whisper-based approach and whether overfitting is a concern.

### EXP-004: Lightweight Custom Audio Model (from scratch)
- **Status**: 🔲 Not Started
- **Hypothesis**: A custom tiny model (~500K params) can achieve competitive accuracy while being significantly faster at inference (<10ms).
- **Architecture**: Mel Spectrogram → 3x Conv1d Blocks → Bidirectional GRU → Attention Pool → Classifier
- **Config**: lr=1e-3, epochs=20, batch_size=64
- **What we learn**: Trade-off between model size, speed, and accuracy for edge/real-time deployment.

### EXP-005: Indian Hinglish & Filler Words Optimization
- **Status**: 🔲 Not Started
- **Hypothesis**: Filtering/upsampling Hindi/Indian-accent samples and evaluating specifically on filler words (midfiller, endfiller) will improve robustness on conversational nuances.
- **Approach**:
  - Filter and upweight Hindi language samples
  - Evaluate on midfiller and endfiller subsets
  - Optimize classification threshold for pause vs endpoint
- **What we learn**: Impact of language-specific data weighting on filler word handling.

### EXP-006: Data Augmentation & Noise Robustness
- **Status**: 🔲 Not Started
- **Hypothesis**: Adding background noise, speed perturbation, and pitch shifting will improve real-world robustness.
- **Augmentations**:
  - Speed perturbation (0.9x - 1.1x)
  - Pitch shifting (±2 semitones)
  - Background noise addition (SNR 10-20dB)
  - Time masking (SpecAugment style)
- **What we learn**: Which augmentations help most for conversational turn detection.

### EXP-007: Threshold Optimization & Calibration
- **Status**: 🔲 Not Started
- **Hypothesis**: The default 0.5 threshold may not be optimal; tuning it on validation set can improve F1 and balance false interruptions vs latency.
- **Approach**: Use best model, sweep thresholds from 0.3 to 0.7, optimize for balanced F1 / user experience.
- **What we learn**: Optimal operating point for voice agents.

### EXP-008: ONNX Export & INT8 Quantization Benchmark
- **Status**: 🔲 Not Started
- **Hypothesis**: Exporting to ONNX and quantizing to INT8 will reduce model size by ~4x (from ~35MB to ~8MB) with minimal (<1%) accuracy loss.
- **Approach**: Dynamic / Static INT8 ONNX quantization, benchmark CPU & GPU latency.
- **What we learn**: Production deployment readiness and speed-up.

## Experiment Dependency Graph
```
EXP-001 (baseline)
  ├── EXP-002 (partial fine-tune)
  │     └── EXP-003 (full fine-tune)
  ├── EXP-004 (lightweight model)
  ├── EXP-005 (Hinglish & filler word focus)
  ├── EXP-006 (augmentation & noise)
  └── Best model from above
        ├── EXP-007 (threshold tuning)
        └── EXP-008 (ONNX INT8 quantization)
```

## Key Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-21 | Use Whisper Tiny encoder backbone | Recommended for turn detection, strong speech representations, compact (~37M params) |
| 2026-08-21 | Use attention pooling over sequence | Captures temporal prosodic cues at utterance end better than simple mean pooling |
| 2026-08-21 | Binary classification with dynamic pos_weight | Handles batch label imbalance smoothly |
| 2026-08-21 | 8-second window with left-padding | Balances context history with computational efficiency |

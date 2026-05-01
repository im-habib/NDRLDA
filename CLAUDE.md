# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

NeuroDRL-Driver-Alerts (NDRLDA): Deep RL framework for adaptive driver vigilance monitoring on the SEED-VIG dataset (SJTU). Frames driver alerting as sequential decision-making (PPO actor-critic), not supervised drowsiness detection. 5-level discrete intervention: No Alert / Soft Visual / Audio / Strong Haptic+Audio / Emergency Rest.

## Python Environment

Virtual environment in `penv/` (Python 3.14.3). **Always activate first**:

```bash
source penv/bin/activate
```

## Commands

```bash
source penv/bin/activate
pip install -r requirements.txt

# Training
python train.py                                          # default config
python train.py --timesteps 1000000 --encoder attention  # override
python train.py --temporal transformer --use-pca         # architecture experiments

# Evaluation
python evaluate.py                          # evaluate on test subjects
python evaluate.py --benchmark-latency      # include latency benchmark
python evaluate.py --model path/to/model.zip

# Real-time simulation
python simulate.py --subject 5              # simulate on subject 5
python simulate.py --export-onnx --analyze  # export + deployment analysis
```

No test suite, lint config, or CI defined yet.

## Architecture

Seven-layer modular pipeline — each layer in its own package:

```
config/settings.py          — All hyperparameters as dataclasses (Config master class)
preprocessing/
  data_loader.py            — SEED-VIG .mat loading, subject-wise, multimodal fusion
  feature_processor.py      — Sliding window, normalization (z-score/minmax/robust), LDS/MA smoothing
  feature_engineer.py       — Optional PCA and autoencoder dimensionality reduction
environment/
  vigilance_env.py          — Custom Gymnasium env (5 actions, PERCLOS-modulated transitions)
  reward.py                 — Composite reward: safety + comfort + efficiency + trend + alert fatigue
models/
  encoders.py               — MLPEncoder, CNNEncoder, AttentionEncoder (swappable via config)
  temporal.py               — LSTMTemporal, TransformerTemporal (swappable via config)
  actor_critic.py           — SB3-compatible ActorCritic with custom VigilanceFeatureExtractor
  ppo_agent.py              — PPO agent factory using Stable-Baselines3
training/
  trainer.py                — Full pipeline: data → env → PPO → train with callbacks
  callbacks.py              — VigilanceCallback (metrics logging, checkpointing), EarlyStoppingCallback
evaluation/
  metrics.py                — 9 metrics: VI, CR, AE, FAR, IF, SIF, DTR, FRE, MTID
  evaluator.py              — Subject-independent cross-validation evaluator
  latency.py                — Inference latency benchmarking (mean/p95/p99/throughput)
visualization/plots.py      — Trajectory plots, reward curves, intervention heatmaps, summary charts
explainability/explainer.py — Permutation feature importance, attention extraction, decision explanations
deployment/
  optimize.py               — ONNX export, INT8 quantization, deployment analysis
  realtime.py               — Real-time streaming simulation pipeline
```

Entry points: `train.py`, `evaluate.py`, `simulate.py`

## Key Design Decisions

- **`environment/` not `env/`**: `.gitignore` ignores `env/` globally — the environment module uses `environment/` to avoid conflict.
- **Config is king**: `config/settings.py` centralizes all hyperparameters as dataclasses. CLI args in entry scripts override config fields. Never hardcode values in modules.
- **Subject-independent evaluation**: Train/test split is by subject ID, not random. `Config.training.test_subjects` controls which subjects are held out.
- **Swappable architectures**: Encoder (mlp/cnn/attention) and temporal module (lstm/transformer) are selected via config strings and factory functions.
- **Reward engineering**: 6-component reward with temporal smoothing. Mathematical formulation documented in `environment/reward.py` docstring.
- **SB3 integration**: Custom `VigilanceFeatureExtractor` plugs into SB3's PPO via `policy_kwargs`. Standalone `VigilanceActorCritic` also available for custom training loops.

## Data

SEED-VIG `.mat` files go in `data/` (gitignored). Must request from https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html. Features: DE/PSD across 5 bands, 17+4 EEG channels, 36-dim EOG. Ground truth: continuous PERCLOS ∈ [0, 1].

## Stack

Python 3.14 | PyTorch | Stable-Baselines3 | Gymnasium | NumPy | SciPy | scikit-learn | Matplotlib | ONNX/ONNXRuntime | SHAP | TensorBoard

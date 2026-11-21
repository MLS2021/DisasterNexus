# DisasterNexus

**DisasterNexus: Bridging Multimodal Semantics and High-Resolution Bitemporal Representations for Unified Disaster Understanding**

DisasterNexus is a unified multimodal framework for bitemporal disaster interpretation across pre- and post-event optical and SAR imagery. A single checkpoint jointly handles nine DisasterM3 tasks:

- **Recognition & Reasoning**: Disaster Scene Recognition (DSR), Disaster Type Recognition (DTR), Disaster-Bearing Body Recognition (BBR), Building Damage Counting (BDC), Road Damage Estimation (DRE), Object Relational Reasoning (ORR)
- **Referring Segmentation**: Pixel-level localization of instruction-specified affected regions (Optical--Optical and Optical--SAR)
- **Long-form Generation**: Disaster Report Generation and Restoration Advice Generation

## Architecture Overview

DisasterNexus pairs two complementary pathways:
1. **Semantic Vision--Language Branch**: Qwen3-VL (4B or 8B) with LoRA adaptation on `q_proj` and `v_proj` (`r=8, alpha=16`) processes `512x512` bitemporal imagery and natural-language instructions.
2. **High-Resolution Bitemporal Spatial Branch**: A frozen Swin-B encoder extracts multiscale pre- and post-event features (`1024x1024`), followed by sensor-conditioned adapters, residual bitemporal fusion, a multiscale deformable-attention pixel decoder, and a 32-query masked transformer decoder. A `<seg>` representation from the language model scores candidate masks to select the final target region.

## Repository Structure

```text
DisasterNexus/
├── configs/                  # Training and model configs (4B & 8B)
├── disasternexus/
│   ├── data/                 # Dataset loading, event split, sampling, collation
│   ├── models/               # Spatial backbone, fusion, decoders, option heads, unified model
│   ├── losses.py             # Joint text, option-head, mask, and attention losses
│   └── metrics.py            # Exact match, cIoU, mIoU, non-empty mIoU, empty accuracy
├── scripts/
│   ├── train.py              # Multi-stage joint training & long-form continuation
│   ├── evaluate.py           # Benchmark evaluation (QA, RefSeg, long-form)
│   └── infer.py              # Single-sample / batch inference CLI
└── tests/
    └── test_pipeline.py      # End-to-end architecture & loss unit tests
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### 1. Training
```bash
python scripts/train.py --config configs/disaster_nexus_4b.yaml
```

### 2. Evaluation
```bash
python scripts/evaluate.py \
  --config configs/disaster_nexus_4b.yaml \
  --checkpoint outputs/disaster_nexus_4b/last.pt \
  --task all
```

### 3. Inference
```bash
python scripts/infer.py \
  --config configs/disaster_nexus_4b.yaml \
  --checkpoint outputs/disaster_nexus_4b/last.pt \
  --pre-image path/to/pre.png \
  --post-image path/to/post.png \
  --modality optical \
  --prompt "Segment the damaged buildings in the post-event image."
```

# Multi-Modal Face Anti-Spoofing for Banking/KYC

A robust face anti-spoofing (FAS) system designed for Banking/KYC verification. Detects print attacks, replay attacks, 3D masks, and deepfakes using multi-modal inputs (RGB, Depth, IR).

## Features

- **Multi-Modal Support**: RGB, Depth, and IR modalities
- **ISO/IEC 30107-3 Compliant Metrics**: APCER, BPCER, ACER, HTER, EER
- **Dataset Support**: CASIA-SURF, OULU-NPU, Replay-Attack
- **Face Detection**: MediaPipe-based face detection and alignment
- **Config-Driven**: YAML-based experiment configuration

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Facial

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── configs/                    # Experiment configurations
│   └── baseline_rgb.yaml       # RGB baseline config
├── data/
│   ├── datasets/               # Dataset storage (gitignored)
│   ├── preprocessing/          # Face detection & alignment
│   │   └── face_detector.py
│   ├── dataset.py              # FASDataset class
│   ├── transforms.py           # Albumentations transforms
│   └── dataloader.py           # DataLoader factory
├── models/
│   ├── backbones/              # Feature extractors
│   ├── heads/                  # Classification heads
│   └── fusion/                 # Multi-modal fusion
├── utils/
│   ├── config.py               # Config loading utilities
│   ├── seed.py                 # Reproducibility
│   ├── metrics.py              # FAS-specific metrics
│   └── logging.py              # Experiment logging
├── trainers/                   # Training loops
├── evaluation/                 # Evaluation scripts
├── scripts/                    # Utility scripts
├── tests/                      # Unit tests
├── checkpoints/                # Model checkpoints (gitignored)
├── experiments/                # Experiment outputs (gitignored)
└── logs/                       # Training logs (gitignored)
```

## Quick Start

### 1. Prepare Dataset

Download and extract CASIA-SURF dataset to `data/datasets/casia_surf/`:

```
data/datasets/casia_surf/
├── Training/
│   ├── real/{subject_id}/{sample}_rgb.jpg
│   └── fake/{subject_id}/{sample}_rgb.jpg
├── Val/
└── Test/
```

### 2. Run Tests

```bash
python -m tests.test_pipeline
```

### 3. Train Model

```bash
python scripts/train.py --config configs/baseline_rgb.yaml
```

## Metrics

Following ISO/IEC 30107-3 standard:

| Metric | Description |
|--------|-------------|
| **APCER** | Attack Presentation Classification Error Rate |
| **BPCER** | Bonafide Presentation Classification Error Rate |
| **ACER** | Average Classification Error Rate: (APCER + BPCER) / 2 |
| **HTER** | Half Total Error Rate |
| **EER** | Equal Error Rate (where FPR = FNR) |

## Label Convention

- `0` = Spoof/Attack
- `1` = Real/Bonafide

## Configuration

Edit `configs/baseline_rgb.yaml` to customize:

```yaml
model:
  backbone: "resnet18"  # resnet18, resnet50, efficientnet_b0
  pretrained: true

data:
  modalities: ["rgb"]   # ["rgb", "depth", "ir"]
  batch_size: 32
  img_size: 224

training:
  epochs: 50
  optimizer:
    lr: 0.001
```

## Tech Stack

- **PyTorch 2.0+**: Deep learning framework
- **timm**: Pretrained backbones
- **albumentations**: Image augmentations
- **MediaPipe**: Face detection
- **scikit-learn**: Metrics computation
- **TensorBoard/W&B**: Experiment tracking

## License

MIT License

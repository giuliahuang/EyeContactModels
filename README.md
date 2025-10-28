# FSFNets: Adaptive Feature Selection and Fusion for Eye Contact Detection

A comprehensive deep learning framework for eye contact detection in group interaction scenarios, featuring FSFNet (Feature Selection and Fusion Network) with ensemble capabilities using traditional machine learning models.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Components](#model-components)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)

## Overview

FSFNets is a state-of-the-art solution for eye contact detection that combines: 

- **FSFNet**: A transformer-based neural network with adaptive feature selection
- **Traditional ML Models**: Support Vector Machines (SVM) and XGBoost for ensemble learning
- **Multi-modal Features**: Visual features from face crops and structured features (gaze, pose, speaker information)

The system addresses the MultiMediate2024 challenge for eye contact detection in group interaction scenarios.

Paper: [DOI](https://doi.org/10.1145/3664647.3688987)

## Architecture

### FSFNet Architecture

```
Input Image (112×112×3)
    ↓
ResNet50 Backbone (pretrained)
    ↓
Feature Maps (196×256)
    ↓
FSFEncoder (Transformer with Adaptive Token Selection)
    ↓
Channel Attention Module
    ↓
Classification Head
    ↓
Output (4 classes)
```

### Key Components

1. **ResNet50 Backbone**: Extracts visual features from face crops
2. **FSFEncoder**: Transformer encoder with adaptive token selection mechanism
3. **Channel Attention**: Enhances feature representation
4. **Classification Head**: Maps features to 4-class eye contact labels

### Model Variants

- **Small**: 4 transformer layers
- **Base**: 6 transformer layers  
- **Large**: 8 transformer layers (default)

### Pretrained Models Organization

```
FSFNet/
├── pretrained/          
├──── ir50.pth         # IR-50 backbone weights for initializing FSFNet
└──── best.pth         # Official FSFNet pretrained checkpoint (baseline)
```

## Dataset

The system uses the MPIIGroupInteraction dataset with the following structure:

### Data Format
- **Images**: Face crops in `.bmp` format (112×112 pixels)
- **Labels**: 4-class eye contact labels (0-3)
- **Features**: Structured features including:
  - Gaze direction vectors (6D)
  - Head pose (6D: translation + rotation)
  - Speaker information (5D one-hot encoding)
  - Speaker status (1D binary)

### Dataset Organization
```
Dataset/
├── frames/           # Face crop images (.bmp)
├── train_val.csv     # Training/validation split with features
└── test.csv         # Test set with features
```

### CSV Column Definitions (`train_val.csv`, `test.csv`)

- **sample_index**: Unique sample identifier (matches image naming)
- **ec_relative**: Eye contact label (0–3)
- **gaze features**: Two 3D gaze vectors (gaze_0, gaze_1) and aggregated angles (gaze_angle_x, gaze_angle_y)
- **head pose**: Translation (pose_Tx, pose_Ty, pose_Tz) and rotation (pose_Rx, pose_Ry, pose_Rz)
- **speaker**: Speaker ID for the current frame
- **subject_pos**: Original subject position index.
- **adjusted_pos**: Aligned subject position index consistent across all recordings.
- **is_speaker**: 1 if the subject is currently speaking, else 0
- **data_subset**: Split tag for the row (e.g., train/val/test)
- **speaker_x**: One-hot speaker identity vector

## Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)
- PyTorch 1.8+

### Dependencies
```bash
pip install torch torchvision
pip install numpy pandas scikit-learn
pip install opencv-python pillow
pip install xgboost joblib
pip install einops
```

### Setup
```bash
git clone https://github.com/giuliahuang/FSFNets.git
cd FSFNets
```

## Usage

### Quick Start

1. **Prepare Dataset**: Place your dataset in the `Dataset/` directory
2. **Prepare Models**: Place pretrained models in the `FSFNet/models/pretrained` directory
3. **Train FSFNet**:
   ```bash
   cd FSFNet
   python train.py
   ```
4. **Evaluate Model**:
   ```bash
   python test.py
   ```

### Training FSFNet

```bash
python train.py --dataset ec -c ./models/pretrained/best.pth --modeltype large --batch_size 128 --epochs 300 --lr 0.000004 --optimizer adam --gpu 1
```

### Training ML Models

```bash
cd MLmodels
python run.py
```

### Evaluation

The evaluation script supports:
- FSFNet standalone evaluation
- ML model evaluation
- Ensemble evaluation (FSFNet + ML)

```bash
cd FSFNet
python test.py
```

## Model Components

### FSFNet (`models/FSFNet.py`)

**Core Architecture**:
- **Backbone**: ResNet50 with pretrained weights
- **FSFEncoder**: Transformer with adaptive token selection
- **CA_Module**: Channel attention mechanism
- **ClassificationHead**: Final classification layer

**Key Features**:
- Adaptive token selection during training
- Sharpness-Aware Minimization (SAM) optimizer
- Random augmentation for data diversity

### FSFEncoder (`models/FSFEncoder.py`)

**Transformer Components**:
- **Multi-head Self-Attention**: With adaptive token pruning
- **MLP Blocks**: Feed-forward networks with GELU activation
- **Layer Normalization**: Applied before attention and MLP
- **Drop Path**: Stochastic depth for regularization

**Adaptive Selection**:
- Dynamic token pruning based on attention scores
- Configurable keep rates for different layers
- Maintains class token throughout the process

### Dataset Handler (`dataset/dataset_ec.py`)

**Features**:
- Multi-modal data loading (images + structured features)
- Flexible feature set selection
- Automatic train/validation splitting
- Data augmentation support

**Feature Sets**:
- `gaze_of`: Gaze direction vectors
- `pose_of`: Head pose (translation + rotation)
- `speaker_info`: Speaker identification
- `is_speaker_info`: Speaker status

## Training


### FSFNet Training

**Optimization**:
- **Optimizer**: SAM (Sharpness-Aware Minimization)
- **Base Optimizer**: Adam/AdamW/SGD
- **Scheduler**: Exponential LR decay (γ=0.98)
- **Loss**: Cross-entropy

**Data Augmentation**:
- Random augmentation (RandAugment)
- Random erasing
- Standard ImageNet normalization

**Training Configuration**:
```python
# Default hyperparameters
batch_size = 128
learning_rate = 0.000004
epochs = 300
optimizer = "adam"
model_type = "large"
```


### ML Model Training

**Algorithms**:
- **SVM**: RBF kernel with grid search
- **XGBoost**: GPU-accelerated with hyperparameter tuning

**Feature Engineering**:
- Standard scaling
- Class balancing
- Feature set combinations

## Evaluation

### Metrics

The system uses a composite scoring metric:
```
Score = 0.67 × Macro F1 + 0.33 × Accuracy
```

### Evaluation Modes

1. **FSFNet Only**: Pure deep learning approach
2. **ML Only**: Traditional machine learning
3. **Ensemble**: Weighted combination of FSFNet and ML predictions

## Results

### Model Performance

The system achieves state-of-the-art performance on eye contact detection:

- **FSFNet**: High accuracy with transformer-based feature learning
- **Ensemble**: Improved robustness through multi-modal fusion
- **Adaptive Selection**: Efficient computation with maintained performance

### Pretrained Models

Available model checkpoints:
- `models/pretrained/ir50.pth`: pretrained model
- `models/pretrained/best.pth`: Base FSFNet model

## 🔬 Technical Details

### Adaptive Token Selection

The FSFEncoder implements dynamic token pruning:
1. Compute attention scores for each token
2. Select top-k tokens based on attention
3. Prune less important tokens
4. Maintain class token throughout

### Sharpness-Aware Minimization

SAM optimizer improves generalization:
1. First forward pass: compute gradients
2. Perturb parameters in gradient direction
3. Second forward pass: compute perturbed gradients
4. Update parameters using perturbed gradients

### Multi-modal Fusion

The system combines:
- **Visual Features**: Extracted by FSFNet from face crops
- **Structured Features**: Gaze, pose, and speaker information
- **Ensemble Prediction**: Weighted combination of predictions

## Citation

```bibtex
@inproceedings{ma2024less,
  title={Less is More: Adaptive Feature Selection and Fusion for Eye Contact Detection},
  author={Ma, Fuyan and He, Yiran and Sun, Bin and Li, Shutao},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  year={2024}
}

@inproceedings{mueller2018eyecontact,
    author = {M\"{u}ller, Philipp and Huang, Michael Xuelin and Zhang, Xucong and Bulling, Andreas},
    title = {Robust eye contact detection in natural multi-person interactions using gaze and speaking behaviour},
    year = {2018},
    booktitle = {Proceedings of the 2018 ACM Symposium on Eye Tracking Research \& Applications}
}
```

## Project Structure

```
FSFNets/
├── FSFNet/                 # Main FSFNet implementation
│   ├── models/            # Model architectures
│   ├── dataset/           # Data loading and preprocessing
│   ├── utils/             # Utility functions
│   ├── train.py           # Training script
│   └── test.py            # Evaluation script
├── FSFNetNP/              # Non-pytorch version
├── MLmodels/              # Traditional ML models
├── Dataset/               # Dataset directory
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Note**: This implementation is part of the MultiMediate2024 challenge solution. For the complete challenge setup and dataset, please refer to the [MultiMediate Challenge](https://multimediate-challenge.org/) website.
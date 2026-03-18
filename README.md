# Multimodal Brain Encoding for Naturalistic Movies

**From Baseline Linear Models to Diversity-Weighted Ensembles**

This repository contains the code for predicting whole-brain fMRI responses to naturalistic multimodal movie stimuli, following the [Algonauts 2025](https://algonauts.csail.mit.edu/) challenge setup.

## Overview

The system combines four complementary architecture families into a parcel-specific weighted ensemble:

| Architecture | Description | Parameters |
|---|---|---|
| **TRIBE-style Encoder** | Transformer-based multimodal fusion with per-modality projections, type embeddings, and bidirectional self-attention | d_model=256, 8 heads, 3 layers |
| **Deep Linear Branch** | Lightweight stacked feedforward layers with batch normalization | hidden_dim=384 |
| **WideLinear Branch** | Wide nonlinear projection with GELU activation | hidden_dim=2048→1024 |
| **Ridge Anchor** | Ridge regression baseline for low-variance anchoring | α=1000 |

Key techniques:
- **Modality Dropout** (p=0.2): Prevents over-reliance on any single modality
- **Mixed Loss**: λ_mse(0.03) · MSE + λ_pearson(1.0) · Pearson correlation
- **Parcel-Specific Weighting**: Temperature-scaled softmax (T=0.3) across ensemble members per brain parcel
- **Early Stopping**: Patience=7 on validation loss with 3-epoch full-data fine-tuning


## Setup

**1.** Create a conda environment:

```bash
conda create -n algonauts python=3.12 -y
conda activate algonauts
pip install -r requirements.txt
```

**2.** Set the path to the Algonauts dataset:

```bash
export DATAPATH="/path/to/algonauts/dataset"
```

## Training

### Using Notebooks (Recommended for Colab)

The primary workflow uses Jupyter notebooks on Google Colab with an NVIDIA L4 GPU:

```bash
# Open notebooks/new_model.ipynb on Google Colab
# Run all cells sequentially
```

### Using the Python Package

```python
from algonauts_brain_encoding.train import (
    train_models_all_subjects,
    generate_friends_s7_submission,
    generate_ood_submission,
)

# Train 20-member ensemble for all 4 subjects
trained_models = train_models_all_subjects(
    root_data_dir="path/to/data",
    modality='all',
    n_ensemble=20,
)

# Generate challenge submissions
generate_friends_s7_submission(trained_models, root_data_dir, "output/")
generate_ood_submission(trained_models, root_data_dir, "output/")
```

## Feature Extraction

Three pretrained models extract per-TR stimulus representations:

| Modality | Model | Dimension | Method |
|---|---|---|---|
| Visual | SlowFusion ResNet-50 (slow_r50) | 250-d per TR | 8 uniformly sampled frames → final pooling layer |
| Audio | librosa MFCCs | 20-d per TR | 20 Mel-frequency cepstral coefficients, averaged per TR |
| Language | BERT-base-uncased | 250-d per TR | Pooler output from context window of up to 510 tokens |

Temporal context: visual and audio features use a sliding window of 5 TRs (total input: 1,250 + 100 + 250 = 1,600 dimensions).

## Results

Architecture-level validation performance (mean Pearson r across 1,000 parcels):

| Architecture | Count | Mean r | Max r |
|---|---|---|---|
| WideLinear | 6 | 0.4287 | 0.4423 |
| TRIBE | 7 | 0.2896 | 0.3011 |
| Deep Linear | 6 | 0.2488 | 0.2497 |
| Ridge | 1 | 0.2658 | 0.2658 |

## References

- [1] TRIBE: TRImodal Brain Encoder ([arXiv:2507.22229](https://arxiv.org/abs/2507.22229))
- [2] VIBE: Video-Input Brain Encoder ([arXiv:2507.17958](https://arxiv.org/abs/2507.17958))
- [3] Multimodal Recurrent Ensembles ([arXiv:2507.17897](https://arxiv.org/abs/2507.17897))
- [9] Algonauts 2025 Challenge ([arXiv:2501.00504](https://arxiv.org/abs/2501.00504))

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

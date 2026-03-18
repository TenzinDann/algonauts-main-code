"""Multimodal Brain Encoding for Naturalistic Movies.

A diversity-weighted ensemble system for predicting whole-brain fMRI responses
to multimodal movie stimuli, following the Algonauts 2025 challenge setup.
"""

from algonauts_brain_encoding.models import (
    TRIBEEncoder,
    MedARCEncoder,
    WideLinearEncoder,
    ModalityDropout,
)
from algonauts_brain_encoding.losses import PearsonLoss, CombinedLoss
from algonauts_brain_encoding.ensemble import (
    BrainPredictorWrapper,
    ParcelSpecificEnsemble,
    RidgeWrapper,
    ScaledPredictor,
    train_ensemble_models,
)
from algonauts_brain_encoding.dataset import AlgonautsDataset

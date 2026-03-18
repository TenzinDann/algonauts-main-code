"""PyTorch dataset for Algonauts brain encoding."""

import torch
from torch.utils.data import Dataset


class AlgonautsDataset(Dataset):
    """Dataset with optional Gaussian noise augmentation.

    Parameters
    ----------
    features : array-like
        Input stimulus features, shape (n_samples, n_features).
    fmri : array-like or None
        Target fMRI responses, shape (n_samples, n_parcels).
        If None, only features are returned (for inference).
    noise_std : float
        Standard deviation of Gaussian noise added during training (default: 0).
    """

    def __init__(self, features, fmri=None, noise_std=0.0):
        self.features = torch.FloatTensor(features)
        self.fmri = torch.FloatTensor(fmri) if fmri is not None else None
        self.noise_std = noise_std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.noise_std > 0 and self.fmri is not None:
            x = x + torch.randn_like(x) * self.noise_std
        if self.fmri is not None:
            return x, self.fmri[idx]
        return x

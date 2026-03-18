"""Ensemble training and parcel-specific weighting for brain encoding.

Implements a diversity-weighted ensemble combining multiple architecture
families (TRIBE Transformer, Deep Linear, WideLinear, Ridge) with
parcel-specific softmax weighting based on validation correlations.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from tqdm import tqdm

from algonauts_brain_encoding.dataset import AlgonautsDataset
from algonauts_brain_encoding.losses import CombinedLoss
from algonauts_brain_encoding.models import (
    MedARCEncoder,
    TRIBEEncoder,
    WideLinearEncoder,
)


class BrainPredictorWrapper:
    """Training wrapper for any encoder architecture.

    Handles the full training loop including mixed-precision training,
    early stopping, cosine annealing with warmup, and final fine-tuning.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    output_dim : int
        Number of brain parcels (default: 1000).
    device : str
        Device for training ('cuda' or 'cpu').
    epochs : int
        Maximum training epochs (default: 60).
    lr : float
        Learning rate (default: 3e-4).
    model_type : str
        Architecture type: 'tribe', 'medarc', or 'wide'.
    modality_dims : list of (int, int) or None
        Modality index ranges for ModalityDropout.
    noise_std : float
        Gaussian noise std for data augmentation (default: 0.01).
    batch_size : int
        Training batch size (default: 256).
    stimulus_window : int
        Temporal context window size (default: 5).
    vis_base, aud_base, lang_base : int
        Per-TR feature dimensions per modality.
    """

    def __init__(self, input_dim, output_dim=1000, device='cuda',
                 epochs=60, lr=3e-4, model_type='tribe', modality_dims=None,
                 noise_std=0.01, batch_size=256,
                 stimulus_window=5, vis_base=250, aud_base=20, lang_base=250):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.epochs = epochs
        self.model_type = model_type
        self.noise_std = noise_std
        self.batch_size = batch_size

        if model_type == 'medarc':
            self.model = MedARCEncoder(
                input_dim, output_dim, hidden_dim=384, dropout=0.3,
                modality_dims=modality_dims).to(self.device)
        elif model_type == 'wide':
            self.model = WideLinearEncoder(
                input_dim, output_dim, hidden_dim=2048, dropout=0.4,
                modality_dims=modality_dims).to(self.device)
        else:  # 'tribe'
            self.model = TRIBEEncoder(
                input_dim, output_dim, d_model=256, nhead=8,
                num_layers=3, dropout=0.3,
                modality_dims=modality_dims,
                stimulus_window=stimulus_window,
                vis_base=vis_base, aud_base=aud_base,
                lang_base=lang_base).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=0.01, betas=(0.9, 0.999))
        self.criterion = CombinedLoss(lambda_mse=0.03, lambda_pearson=1.0)
        self.scaler = (torch.amp.GradScaler('cuda')
                       if self.device == 'cuda' else None)

    def fit(self, X, y):
        """Train the model with early stopping and final fine-tuning.

        Uses an internal 95/5 train/val split for early stopping (patience=7),
        then fine-tunes on full data for 3 epochs.
        """
        n = len(X)
        split = int(n * 0.95)
        X_train_inner, y_train_inner = X[:split], y[:split]
        X_val_inner, y_val_inner = X[split:], y[split:]

        train_dataset = AlgonautsDataset(
            X_train_inner, y_train_inner, noise_std=self.noise_std)
        loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=(self.device == 'cuda'))

        val_X_tensor = torch.FloatTensor(X_val_inner).to(self.device)
        val_y_tensor = torch.FloatTensor(y_val_inner).to(self.device)

        warmup_epochs = max(1, self.epochs // 10)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs - warmup_epochs)

        best_val_loss = float('inf')
        best_state = None
        patience, patience_counter = 7, 0

        self.model.train()
        pbar = tqdm(range(self.epochs),
                    desc=f"  {self.model_type.upper()}", leave=True)

        for epoch in pbar:
            total_loss, n_batches = 0, 0
            self.model.train()
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()

                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        loss = self.criterion(
                            self.model(batch_X), batch_y)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.criterion(self.model(batch_X), batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if epoch >= warmup_epochs:
                scheduler.step()

            train_loss = total_loss / n_batches

            # Validation-based early stopping
            self.model.eval()
            with torch.no_grad():
                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        val_pred = self.model(val_X_tensor)
                        val_loss = self.criterion(
                            val_pred, val_y_tensor).item()
                else:
                    val_pred = self.model(val_X_tensor)
                    val_loss = self.criterion(
                        val_pred, val_y_tensor).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            pbar.set_postfix({
                'TrL': f"{train_loss:.4f}",
                'VaL': f"{val_loss:.4f}",
                'Best': f"{best_val_loss:.4f}",
                'P': f"{patience_counter}/{patience}"
            })

            if patience_counter >= patience:
                pbar.write(f"    Early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Fine-tune on full data for 3 epochs
        self.model.train()
        for pg in self.optimizer.param_groups:
            pg['lr'] = 1e-4
        full_dataset = AlgonautsDataset(X, y, noise_std=0)
        full_loader = DataLoader(
            full_dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(3):
            for batch_X, batch_y in full_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        loss = self.criterion(
                            self.model(batch_X), batch_y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self.criterion(self.model(batch_X), batch_y)
                    loss.backward()
                    self.optimizer.step()

        return self

    def predict(self, X):
        """Generate predictions in eval mode."""
        self.model.eval()
        dataset = AlgonautsDataset(X)
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch_X in loader:
                if self.device == 'cuda':
                    with torch.amp.autocast('cuda'):
                        out = self.model(batch_X.to(self.device))
                else:
                    out = self.model(batch_X.to(self.device))
                preds.append(out.float().cpu().numpy())
        return np.vstack(preds)


class ParcelSpecificEnsemble:
    """Parcel-specific weighted ensemble using temperature-scaled softmax.

    For each of the 1,000 brain parcels, computes independent weights across
    ensemble members based on their validation Pearson correlation, then
    applies temperature-scaled softmax to obtain per-parcel weights.

    Parameters
    ----------
    models : list
        List of trained model objects (each must have .predict() and .model_type).
    temperature : float
        Softmax temperature for weight sharpening (default: 0.1).
    """

    def __init__(self, models, temperature=0.1):
        self.models = models
        self.temperature = temperature
        self.weights = None
        self.ensemble_val_r = None

    def fit_weights(self, X_val, y_val):
        """Learn per-parcel ensemble weights from validation data."""
        n_models = len(self.models)
        n_parcels = y_val.shape[1]
        scores = np.zeros((n_models, n_parcels))

        print(f"  Computing per-parcel ensemble weights "
              f"for {n_models} models...")
        for m, model in enumerate(self.models):
            pred = model.predict(X_val)
            for p in range(n_parcels):
                r, _ = pearsonr(y_val[:, p], pred[:, p])
                scores[m, p] = max(r, 0)

        # Report per-model performance
        for m, model in enumerate(self.models):
            mean_r = scores[m].mean()
            print(f"    Model {m} ({model.model_type}): "
                  f"mean r = {mean_r:.4f}")

        # Temperature-scaled softmax
        scores_scaled = scores / self.temperature
        exp_scores = np.exp(
            scores_scaled - scores_scaled.max(axis=0, keepdims=True))
        self.weights = exp_scores / (
            exp_scores.sum(axis=0, keepdims=True) + 1e-8)

        print(f"  Mean max weight: "
              f"{np.mean(np.max(self.weights, axis=0)):.4f}")

        # Evaluate weighted ensemble on validation set
        all_preds = np.stack(
            [model.predict(X_val) for model in self.models])
        ensemble_pred = (
            all_preds * self.weights[:, np.newaxis, :]).sum(axis=0)
        ensemble_parcels_r = np.array([
            pearsonr(y_val[:, p], ensemble_pred[:, p])[0]
            for p in range(n_parcels)])
        ensemble_mean_r = np.mean(np.clip(ensemble_parcels_r, 0, None))
        best_single = scores.mean(axis=1).max()

        print(f"\n  >>> Ensemble validation: mean r = {ensemble_mean_r:.4f}"
              f" (best single model: {best_single:.4f}, "
              f"improvement: +{ensemble_mean_r - best_single:.4f}) <<<")
        self.ensemble_val_r = ensemble_mean_r

        return self

    def predict(self, X):
        """Generate weighted ensemble predictions."""
        preds = np.stack([m.predict(X) for m in self.models])
        if self.weights is not None:
            return (preds * self.weights[:, np.newaxis, :]).sum(axis=0)
        return preds.mean(axis=0)


class RidgeWrapper:
    """Sklearn Ridge regression wrapped for ensemble compatibility.

    Parameters
    ----------
    alpha : float
        Ridge regularization strength (default: 1000).
    """

    def __init__(self, alpha=1000):
        self.model_type = 'ridge'
        self.ridge = Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, X, y):
        self.ridge.fit(X, y)
        return self

    def predict(self, X):
        return self.ridge.predict(X).astype(np.float32)


class ScaledPredictor:
    """Wrapper that handles feature/fMRI standardization at inference.

    Parameters
    ----------
    model : object
        Trained model with .predict() method.
    feat_scaler : StandardScaler
        Fitted feature scaler.
    fmri_scaler : StandardScaler
        Fitted fMRI scaler.
    """

    def __init__(self, model, feat_scaler, fmri_scaler):
        self.model = model
        self.feat_scaler = feat_scaler
        self.fmri_scaler = fmri_scaler

    def predict(self, X):
        X_scaled = self.feat_scaler.transform(X)
        pred_scaled = self.model.predict(X_scaled)
        return self.fmri_scaler.inverse_transform(pred_scaled)


def train_ensemble_models(features_train, fmri_train, n_models=15,
                          features_val=None, fmri_val=None,
                          modality_dims=None, stimulus_window=5,
                          vis_base=250, aud_base=20, lang_base=250):
    """Train a diverse ensemble of brain encoding models.

    Creates an ensemble of n_models members: 1 Ridge anchor plus neural
    models split equally among TRIBE, MedARC (Deep Linear), and WideLinear
    architectures, each with independently randomized hyperparameters.

    Parameters
    ----------
    features_train : ndarray, shape (n_samples, n_features)
        Training features (already standardized).
    fmri_train : ndarray, shape (n_samples, n_parcels)
        Training fMRI targets (already standardized).
    n_models : int
        Total number of ensemble members (default: 15).
    features_val, fmri_val : ndarray or None
        Validation data for parcel-specific weight fitting.
    modality_dims : list of (int, int) or None
        Modality index ranges.
    stimulus_window : int
        Temporal context window size.
    vis_base, aud_base, lang_base : int
        Per-TR feature dimensions per modality.

    Returns
    -------
    ParcelSpecificEnsemble or single model
        Trained ensemble (or single model if n_models=1).
    """
    input_dim = features_train.shape[1]
    models = []

    # Ridge regression anchor (fast baseline)
    print(f"\n{'='*60}")
    print(f"  Ensemble 1/{n_models} - Ridge regression (fast baseline)")
    print(f"{'='*60}")
    ridge_model = RidgeWrapper(alpha=1000)
    ridge_model.fit(features_train, fmri_train)
    models.append(ridge_model)
    print("  Ridge done.")

    # Allocate remaining slots among TRIBE + MedARC + WideLinear
    n_nn = n_models - 1
    configs = []
    n_per = max(1, n_nn // 3)

    for i in range(n_per + (1 if n_nn % 3 > 0 else 0)):
        configs.append({
            'type': 'tribe',
            'epochs': random.choice([60, 70, 80]),
            'lr': 3e-4 * (0.7 + 0.6 * random.random()),
            'batch_size': random.choice([256, 384]),
            'noise_std': random.choice([0.005, 0.01, 0.02]),
        })

    for i in range(n_per + (1 if n_nn % 3 > 1 else 0)):
        configs.append({
            'type': 'medarc',
            'epochs': random.choice([60, 70, 80]),
            'lr': 3e-4 * (0.7 + 0.6 * random.random()),
            'batch_size': random.choice([256, 384]),
            'noise_std': random.choice([0.005, 0.01, 0.02]),
        })

    for i in range(n_per):
        configs.append({
            'type': 'wide',
            'epochs': random.choice([50, 60, 70]),
            'lr': 3e-4 * (0.7 + 0.6 * random.random()),
            'batch_size': random.choice([256, 384]),
            'noise_std': random.choice([0.005, 0.01, 0.02]),
        })

    configs = configs[:n_nn]
    random.shuffle(configs)

    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"  Ensemble {i+2}/{n_models} "
              f"(seed={42+i}, arch={cfg['type'].upper()})")
        print(f"{'='*60}")
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        random.seed(42 + i)

        wrapper = BrainPredictorWrapper(
            input_dim=input_dim, output_dim=1000,
            epochs=cfg['epochs'], lr=cfg['lr'],
            model_type=cfg['type'], modality_dims=modality_dims,
            noise_std=cfg['noise_std'], batch_size=cfg['batch_size'],
            stimulus_window=stimulus_window,
            vis_base=vis_base, aud_base=aud_base, lang_base=lang_base)
        wrapper.fit(features_train, fmri_train)
        models.append(wrapper)

    if n_models > 1 and features_val is not None and fmri_val is not None:
        ensemble = ParcelSpecificEnsemble(models, temperature=0.3)
        ensemble.fit_weights(features_val, fmri_val)
        return ensemble
    elif n_models > 1:
        return ParcelSpecificEnsemble(models, temperature=0.3)
    else:
        return models[0]

"""Main training and prediction pipeline.

Orchestrates end-to-end training: loads data, trains per-subject ensembles,
generates Friends S7 and OOD submissions.
"""

import gc
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from algonauts_brain_encoding.data_utils import (
    align_features_and_fmri_samples,
    align_features_friends_s7,
    align_features_ood,
    load_fmri,
    load_stimulus_features,
    load_stimulus_features_friends_s7,
    load_stimulus_features_ood,
)
from algonauts_brain_encoding.ensemble import (
    ScaledPredictor,
    train_ensemble_models,
)


def train_models_all_subjects(root_data_dir, modality='all',
                               excluded_samples_start=5,
                               excluded_samples_end=5,
                               hrf_delay=3, stimulus_window=5,
                               n_ensemble=20):
    """Train per-subject ensemble models on all training data.

    Parameters
    ----------
    root_data_dir : str
        Root data directory containing the Algonauts dataset.
    modality : str
        Feature modality: 'visual', 'audio', 'language', or 'all'.
    excluded_samples_start : int
        Number of initial TRs to exclude (default: 5).
    excluded_samples_end : int
        Number of final TRs to exclude (default: 5).
    hrf_delay : int
        HRF delay in TRs (default: 3, ~4.5s).
    stimulus_window : int
        Temporal context window (default: 5).
    n_ensemble : int
        Number of ensemble members per subject (default: 20).

    Returns
    -------
    trained_models : dict
        {subject_id: ScaledPredictor} for each subject.
    """
    trained_models = {}
    subjects = [1, 2, 3, 5]

    print("Loading features...")
    features = load_stimulus_features(root_data_dir, modality)

    # Auto-detect modality dimensions
    mod_dims = None
    v_base = a_base = l_base = 0
    if modality == 'all':
        v_base = features['visual'][
            next(iter(features['visual']))].shape[1]
        a_base = features['audio'][
            next(iter(features['audio']))].shape[1]
        l_base = features['language'][
            next(iter(features['language']))].shape[1]
        vis_dim = v_base * stimulus_window
        aud_dim = a_base * stimulus_window
        mod_dims = [
            (0, vis_dim),
            (vis_dim, vis_dim + aud_dim),
            (vis_dim + aud_dim, vis_dim + aud_dim + l_base),
        ]
        print(f"Modality dims: V={vis_dim}, A={aud_dim}, L={l_base}")

    movies = [
        "friends-s01", "friends-s02", "friends-s03",
        "friends-s04", "friends-s05", "friends-s06",
        "movie10-bourne", "movie10-figures",
        "movie10-life", "movie10-wolf",
    ]

    for s in subjects:
        print(f"\n{'#'*60}")
        print(f"  Training ensemble for sub-0{s} (n={n_ensemble})")
        print(f"{'#'*60}")

        fmri = load_fmri(root_data_dir, s)
        feat_all, fmri_all = align_features_and_fmri_samples(
            features, fmri, excluded_samples_start,
            excluded_samples_end, hrf_delay, stimulus_window, movies)

        print(f"  Data shape: {feat_all.shape}")

        # Standardize
        f_scaler = StandardScaler()
        feat_scaled = f_scaler.fit_transform(feat_all)
        fmri_scaler = StandardScaler()
        fmri_scaled = fmri_scaler.fit_transform(fmri_all)

        # Use 10% random subset for ensemble weight fitting
        n_total = len(feat_scaled)
        val_idx = np.random.RandomState(42).choice(
            n_total, size=int(n_total * 0.1), replace=False)
        X_val = feat_scaled[val_idx]
        y_val = fmri_scaled[val_idx]

        raw_model = train_ensemble_models(
            features_train=feat_scaled,
            fmri_train=fmri_scaled,
            n_models=n_ensemble,
            features_val=X_val,
            fmri_val=y_val,
            modality_dims=mod_dims,
            stimulus_window=stimulus_window,
            vis_base=v_base, aud_base=a_base, lang_base=l_base)

        trained_models[f'sub-0{s}'] = ScaledPredictor(
            raw_model, f_scaler, fmri_scaler)

        del fmri, feat_all, fmri_all, feat_scaled, fmri_scaled
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del features
    gc.collect()
    return trained_models


def generate_friends_s7_submission(trained_models, root_data_dir,
                                    save_dir):
    """Generate Friends S7 submission predictions.

    Parameters
    ----------
    trained_models : dict
        Trained models from train_models_all_subjects().
    root_data_dir : str
        Root data directory.
    save_dir : str
        Output directory for predictions.
    """
    print("Loading Friends S7 features...")
    features_s7 = load_stimulus_features_friends_s7(root_data_dir)

    print("Aligning features...")
    aligned_s7 = align_features_friends_s7(features_s7, root_data_dir)

    submission = {}
    for sub in ['sub-01', 'sub-02', 'sub-03', 'sub-05']:
        submission[sub] = {}
        model = trained_models[sub]
        for epi, feat in aligned_s7[sub].items():
            pred = model.predict(feat)
            submission[sub][epi] = pred.astype(np.float32)
            print(f"  {sub}/{epi}: {pred.shape}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,
                             'fmri_predictions_friends_s7.npy')
    np.save(save_path, submission)
    print(f"Saved to {save_path}")
    return submission


def generate_ood_submission(trained_models, root_data_dir, save_dir):
    """Generate OOD submission predictions.

    Parameters
    ----------
    trained_models : dict
        Trained models from train_models_all_subjects().
    root_data_dir : str
        Root data directory.
    save_dir : str
        Output directory for predictions.
    """
    print("Loading OOD features...")
    features_ood = load_stimulus_features_ood(root_data_dir)

    print("Aligning features...")
    aligned_ood = align_features_ood(features_ood, root_data_dir)

    submission = {}
    for sub in ['sub-01', 'sub-02', 'sub-03', 'sub-05']:
        submission[sub] = {}
        model = trained_models[sub]
        for movie, feat in aligned_ood[sub].items():
            pred = model.predict(feat)
            submission[sub][movie] = pred.astype(np.float32)
            print(f"  {sub}/{movie}: {pred.shape}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'fmri_predictions_ood.npy')
    np.save(save_path, submission)
    print(f"Saved to {save_path}")
    return submission

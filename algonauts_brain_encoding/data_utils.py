"""Data loading, feature extraction, and alignment utilities.

Handles loading of fMRI data, stimulus features, and temporal alignment
between stimulus and brain responses accounting for HRF delay.
"""

import os

import h5py
import numpy as np


def load_fmri(root_data_dir, subject):
    """Load fMRI responses for a given subject.

    Parameters
    ----------
    root_data_dir : str
        Root data directory containing the Algonauts dataset.
    subject : int
        Subject number (1, 2, 3, or 5).

    Returns
    -------
    fmri : dict
        Dictionary mapping movie split names to fMRI arrays
        of shape (n_timepoints, 1000).
    """
    fmri_dir = os.path.join(
        root_data_dir, 'algonauts_2025.competitors', 'fmri',
        f'sub-0{subject}', 'parcellated')
    fmri = {}
    h5_path = os.path.join(fmri_dir,
                           f'sub-0{subject}_fmri_parcellated.h5')
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            fmri[key] = np.array(f[key], dtype=np.float32)
    return fmri


def load_stimulus_features(root_data_dir, modality):
    """Load pre-extracted stimulus features for all modalities.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    modality : str
        'visual', 'audio', 'language', or 'all'.

    Returns
    -------
    features : dict
        If modality='all', returns {mod: {split: array}} for each modality.
        Otherwise returns {split: array} for the selected modality.
    """
    if modality == 'all':
        features = {}
        for mod in ['visual', 'audio', 'language']:
            stimuli_dir = os.path.join(
                root_data_dir, 'stimulus_features', 'pca',
                'friends_movie10', mod, 'features_train.npy')
            features[mod] = np.load(stimuli_dir,
                                    allow_pickle=True).item()
        return features
    else:
        stimuli_dir = os.path.join(
            root_data_dir, 'stimulus_features', 'pca',
            'friends_movie10', modality, 'features_train.npy')
        return np.load(stimuli_dir, allow_pickle=True).item()


def load_stimulus_features_friends_s7(root_data_dir):
    """Load stimulus features for Friends season 7 (test set).

    Parameters
    ----------
    root_data_dir : str
        Root data directory.

    Returns
    -------
    features : dict
        {modality: {episode: array}} for visual, audio, language.
    """
    features = {}
    for mod in ['visual', 'audio', 'language']:
        stimuli_dir = os.path.join(
            root_data_dir, 'stimulus_features', 'pca',
            'friends_movie10', mod, 'features_test.npy')
        features[mod] = np.load(stimuli_dir,
                                allow_pickle=True).item()
    return features


def load_stimulus_features_ood(root_data_dir):
    """Load stimulus features for out-of-distribution movies.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.

    Returns
    -------
    features : dict
        {modality: {movie: array}} for visual, audio, language.
    """
    features = {}
    for mod in ['visual', 'audio', 'language']:
        stimuli_dir = os.path.join(
            root_data_dir, 'stimulus_features', 'pca',
            'ood', mod, 'features_test.npy')
        features[mod] = np.load(stimuli_dir,
                                allow_pickle=True).item()
    return features


def align_features_and_fmri_samples(features, fmri,
                                     excluded_samples_start,
                                     excluded_samples_end,
                                     hrf_delay, stimulus_window, movies):
    """Align stimulus features with fMRI responses for training.

    Creates windowed stimulus features aligned with fMRI BOLD responses,
    accounting for HRF delay. Visual and audio features use a sliding
    window of `stimulus_window` frames; language features use a single
    frame per TR (since BERT embeddings already capture context).

    Parameters
    ----------
    features : dict
        {modality: {split: array}} stimulus features.
    fmri : dict
        {split: array} fMRI responses.
    excluded_samples_start : int
        Number of initial TRs to exclude.
    excluded_samples_end : int
        Number of final TRs to exclude.
    hrf_delay : int
        HRF delay in TRs (~4.5s at TR=1.49s).
    stimulus_window : int
        Number of temporal frames for visual/audio windowing.
    movies : list of str
        Movie names to include.

    Returns
    -------
    aligned_features : ndarray, shape (n_samples, n_features)
    aligned_fmri : ndarray, shape (n_samples, 1000)
    """
    aligned_features = []
    aligned_fmri = np.empty((0, 1000), dtype=np.float32)

    for movie in movies:
        if movie[:7] == 'friends':
            id_ = movie[8:]
        elif movie[:7] == 'movie10':
            id_ = movie[8:]
        else:
            id_ = movie
        movie_splits = [key for key in fmri if id_ in key[:len(id_)]]

        for split in movie_splits:
            fmri_split = fmri[split]
            fmri_split = fmri_split[
                excluded_samples_start:-excluded_samples_end]
            aligned_fmri = np.append(aligned_fmri, fmri_split, 0)

            for s in range(len(fmri_split)):
                f_all = np.empty(0)

                for mod in features.keys():
                    if mod == 'visual' or mod == 'audio':
                        if s < (stimulus_window + hrf_delay):
                            idx_start = excluded_samples_start
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = (s + excluded_samples_start
                                         - hrf_delay - stimulus_window + 1)
                            idx_end = idx_start + stimulus_window
                        if idx_end > len(features[mod][split]):
                            idx_end = len(features[mod][split])
                            idx_start = idx_end - stimulus_window
                        f = features[mod][split][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())

                    elif mod == 'language':
                        if s < hrf_delay:
                            idx = excluded_samples_start
                        else:
                            idx = s + excluded_samples_start - hrf_delay
                        if idx >= (len(features[mod][split]) - hrf_delay):
                            f = features[mod][split][-1, :]
                        else:
                            f = features[mod][split][idx]
                        f_all = np.append(f_all, f.flatten())

                aligned_features.append(f_all)

    aligned_features = np.asarray(aligned_features, dtype=np.float32)
    return aligned_features, aligned_fmri


def align_features_friends_s7(features_friends_s7, root_data_dir,
                               hrf_delay=3, stimulus_window=5):
    """Align stimulus features for Friends S7 test prediction.

    Parameters
    ----------
    features_friends_s7 : dict
        Stimulus features loaded by load_stimulus_features_friends_s7().
    root_data_dir : str
        Root data directory.
    hrf_delay : int
        HRF delay in TRs.
    stimulus_window : int
        Temporal window size.

    Returns
    -------
    aligned : dict
        {subject: {episode: aligned_features_array}}.
    """
    aligned = {}
    subjects = [1, 2, 3, 5]

    for sub in subjects:
        aligned[f'sub-0{sub}'] = {}
        samples_dir = os.path.join(
            root_data_dir, 'algonauts_2025.competitors', 'fmri',
            f'sub-0{sub}', 'target_sample_number',
            f'sub-0{sub}_friends-s7_fmri_samples.npy')
        fmri_samples = np.load(samples_dir, allow_pickle=True).item()

        for epi, samples in fmri_samples.items():
            features_epi = []
            for s in range(samples):
                f_all = np.empty(0)
                for mod in features_friends_s7.keys():
                    if mod in ('visual', 'audio'):
                        if s < (stimulus_window + hrf_delay):
                            idx_start = 0
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = (s - hrf_delay
                                         - stimulus_window + 1)
                            idx_end = idx_start + stimulus_window
                        if idx_end > len(features_friends_s7[mod][epi]):
                            idx_end = len(features_friends_s7[mod][epi])
                            idx_start = idx_end - stimulus_window
                        f = features_friends_s7[mod][epi][
                            idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())
                    elif mod == 'language':
                        if s < hrf_delay:
                            idx = 0
                        else:
                            idx = s - hrf_delay
                        if idx >= (len(features_friends_s7[mod][epi])
                                   - hrf_delay):
                            f = features_friends_s7[mod][epi][-1, :]
                        else:
                            f = features_friends_s7[mod][epi][idx]
                        f_all = np.append(f_all, f.flatten())
                features_epi.append(f_all)

            aligned[f'sub-0{sub}'][epi] = np.asarray(
                features_epi, dtype=np.float32)

    return aligned


def align_features_ood(features_ood, root_data_dir,
                        hrf_delay=3, stimulus_window=5):
    """Align stimulus features for OOD test prediction.

    Parameters
    ----------
    features_ood : dict
        Stimulus features loaded by load_stimulus_features_ood().
    root_data_dir : str
        Root data directory.
    hrf_delay : int
        HRF delay in TRs.
    stimulus_window : int
        Temporal window size.

    Returns
    -------
    aligned : dict
        {subject: {movie: aligned_features_array}}.
    """
    aligned = {}
    subjects = [1, 2, 3, 5]

    for sub in subjects:
        aligned[f'sub-0{sub}'] = {}
        samples_dir = os.path.join(
            root_data_dir, 'algonauts_2025.competitors', 'fmri',
            f'sub-0{sub}', 'target_sample_number',
            f'sub-0{sub}_ood_fmri_samples.npy')
        fmri_samples = np.load(samples_dir, allow_pickle=True).item()

        for movie, samples in fmri_samples.items():
            features_movie = []
            for s in range(samples):
                f_all = np.empty(0)
                for mod in features_ood.keys():
                    if mod in ('visual', 'audio'):
                        if s < (stimulus_window + hrf_delay):
                            idx_start = 0
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = (s - hrf_delay
                                         - stimulus_window + 1)
                            idx_end = idx_start + stimulus_window
                        if idx_end > len(features_ood[mod][movie]):
                            idx_end = len(features_ood[mod][movie])
                            idx_start = idx_end - stimulus_window
                        f = features_ood[mod][movie][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())
                    elif mod == 'language':
                        if s < hrf_delay:
                            idx = 0
                        else:
                            idx = s - hrf_delay
                        if idx >= (len(features_ood[mod][movie])
                                   - hrf_delay):
                            f = features_ood[mod][movie][-1, :]
                        else:
                            f = features_ood[mod][movie][idx]
                        f_all = np.append(f_all, f.flatten())
                features_movie.append(f_all)

            aligned[f'sub-0{sub}'][movie] = np.asarray(
                features_movie, dtype=np.float32)

    return aligned

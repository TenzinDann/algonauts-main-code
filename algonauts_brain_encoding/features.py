"""Feature extraction from pretrained models.

Extracts visual features (SlowFusion ResNet-50), audio features (MFCCs),
and language features (BERT-base-uncased) from movie stimuli.
"""

import os

import librosa
import numpy as np
import torch
from torchvision.models.feature_extraction import create_feature_extractor
from pytorchvideo.transforms import (
    Normalize,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda, CenterCrop
from transformers import BertTokenizer, BertModel


# ---------------------------------------------------------------------------
# Visual features
# ---------------------------------------------------------------------------

def define_frames_transform():
    """Preprocessing pipeline for video frames (specific to slow_r50)."""
    return Compose([
        UniformTemporalSubsample(8),
        Lambda(lambda x: x / 255.0),
        Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        ShortSideScale(size=256),
        CenterCrop(256),
    ])


def get_vision_model(device):
    """Load pre-trained slow_r50 video model for feature extraction.

    Returns
    -------
    feature_extractor : torch.nn.Module
        Feature extractor model.
    model_layer : str
        Layer name for feature extraction.
    transform : callable
        Video frame preprocessing transform.
    """
    model = torch.hub.load(
        'facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    model = model.to(device).eval()
    model_layer = 'blocks.5.pool'
    feature_extractor = create_feature_extractor(
        model, return_nodes=[model_layer])
    transform = define_frames_transform()
    return feature_extractor, model_layer, transform


def extract_visual_features(episode_path, tr, feature_extractor,
                             model_layer, transform, device,
                             save_dir_temp, save_dir_features):
    """Extract visual features from a movie episode.

    Uses SlowFusion ResNet-50 (slow_r50) pretrained on Kinetics-400.
    For each TR, 8 uniformly sampled frames are passed through the
    network, and activations from the final pooling layer yield a
    250-dimensional embedding per TR.

    Parameters
    ----------
    episode_path : str
        Path to the movie file (.mkv).
    tr : float
        Repetition time in seconds (1.49).
    feature_extractor : torch.nn.Module
        Pretrained feature extractor.
    model_layer : str
        Target layer name.
    transform : callable
        Frame preprocessing pipeline.
    device : str
        Computation device.
    save_dir_temp : str
        Temporary directory for intermediate files.
    save_dir_features : str
        Directory to save extracted features.

    Returns
    -------
    features : ndarray
        Visual features of shape (n_trs, feature_dim).
    """
    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(episode_path)
    duration = clip.duration
    n_trs = int(np.floor(duration / tr))

    features_list = []
    for i in range(n_trs):
        start_time = i * tr
        end_time = start_time + tr
        subclip = clip.subclip(start_time, min(end_time, duration))
        frames = []
        for frame in subclip.iter_frames():
            frames.append(frame)
        if len(frames) == 0:
            continue

        frames_tensor = torch.FloatTensor(
            np.array(frames)).permute(3, 0, 1, 2).unsqueeze(0)
        frames_tensor = transform(frames_tensor).to(device)

        with torch.no_grad():
            feat = feature_extractor(frames_tensor)
        feat = feat[model_layer].flatten().cpu().numpy()
        features_list.append(feat)

    clip.close()
    features = np.array(features_list, dtype=np.float32)

    os.makedirs(save_dir_features, exist_ok=True)
    episode_name = os.path.splitext(os.path.basename(episode_path))[0]
    np.save(os.path.join(save_dir_features, f'{episode_name}.npy'),
            features)

    return features


# ---------------------------------------------------------------------------
# Audio features
# ---------------------------------------------------------------------------

def extract_audio_features(episode_path, tr, sr=22050,
                            n_mfcc=20, save_dir_features=None):
    """Extract MFCC audio features from a movie episode.

    Computes 20 Mel-frequency cepstral coefficients (MFCCs) using librosa,
    averaged across time bins within each TR window.

    Parameters
    ----------
    episode_path : str
        Path to the movie file.
    tr : float
        Repetition time in seconds.
    sr : int
        Audio sample rate (default: 22050).
    n_mfcc : int
        Number of MFCCs (default: 20).
    save_dir_features : str or None
        Directory to save features.

    Returns
    -------
    features : ndarray
        Audio features of shape (n_trs, n_mfcc).
    """
    y, _ = librosa.load(episode_path, sr=sr)
    samples_per_tr = int(tr * sr)
    n_trs = len(y) // samples_per_tr

    features_list = []
    for i in range(n_trs):
        segment = y[i * samples_per_tr:(i + 1) * samples_per_tr]
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        features_list.append(mfcc.mean(axis=1))

    features = np.array(features_list, dtype=np.float32)

    if save_dir_features:
        os.makedirs(save_dir_features, exist_ok=True)
        episode_name = os.path.splitext(
            os.path.basename(episode_path))[0]
        np.save(os.path.join(save_dir_features,
                             f'{episode_name}.npy'), features)

    return features


# ---------------------------------------------------------------------------
# Language features
# ---------------------------------------------------------------------------

def get_language_model(device):
    """Load pre-trained BERT-base-uncased model and tokenizer.

    Returns
    -------
    model : BertModel
        BERT model.
    tokenizer : BertTokenizer
        BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to(device).eval()
    return model, tokenizer


def extract_language_features(episode_path, model, tokenizer,
                               num_used_tokens, kept_tokens_last_hidden_state,
                               device, save_dir_features):
    """Extract language features from movie transcripts using BERT.

    For each TR, transcript tokens within a context window of up to
    510 tokens are fed through BERT; the pooler output (768-d) is
    used as the language representation.

    Parameters
    ----------
    episode_path : str
        Path to transcript file (.tsv).
    model : BertModel
        Pretrained BERT model.
    tokenizer : BertTokenizer
        BERT tokenizer.
    num_used_tokens : int
        Maximum context window size.
    kept_tokens_last_hidden_state : int
        Output dimension after reduction.
    device : str
        Computation device.
    save_dir_features : str
        Directory to save features.

    Returns
    -------
    features : ndarray
        Language features of shape (n_trs, feature_dim).
    """
    import pandas as pd

    transcript = pd.read_csv(episode_path, sep='\t')

    features_list = []
    for _, row in transcript.iterrows():
        text = str(row.get('text', ''))
        if not text.strip():
            features_list.append(np.zeros(kept_tokens_last_hidden_state,
                                          dtype=np.float32))
            continue

        inputs = tokenizer(
            text, return_tensors='pt', max_length=num_used_tokens,
            truncation=True, padding='max_length')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        pooler = outputs.pooler_output.cpu().numpy().flatten()
        features_list.append(
            pooler[:kept_tokens_last_hidden_state].astype(np.float32))

    features = np.array(features_list, dtype=np.float32)

    if save_dir_features:
        os.makedirs(save_dir_features, exist_ok=True)
        episode_name = os.path.splitext(
            os.path.basename(episode_path))[0]
        np.save(os.path.join(save_dir_features,
                             f'{episode_name}.npy'), features)

    return features

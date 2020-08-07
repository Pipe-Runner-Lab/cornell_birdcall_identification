from os import (makedirs, path, listdir)
from shutil import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import soundfile as sf
from pathlib import Path
import random
from utils.paths import DATA_ROOT_DIR


def fold_creator(org_dataset_path, dataset_path, number_of_folds):
    if path.exists(dataset_path) == False:
        print("< Fold does not exist, creating new! >")

        # Create fold dirs
        for i in range(number_of_folds):
            makedirs(path.join(dataset_path, str(i)))
            makedirs(path.join(dataset_path, str(i), "train"))
            makedirs(path.join(dataset_path, str(i), "val"))

        data_frame = pd.read_csv(path.join(org_dataset_path, "train.csv"))
        X = data_frame.values
        # one_hot = data_frame.iloc[:, -4:].values
        # Y = [np.where(r == 1)[0][0] for r in one_hot]
        Y = data_frame.iloc[:, -1:].values

        skf = StratifiedKFold(n_splits=number_of_folds)

        fold_idx = 0
        for train_index, val_index in skf.split(X, Y):
            # Copy train csv and images to fold
            for idx in train_index:
                # image_path = str(data_frame.iloc[idx, 0]) + '.jpg'
                image_path = str(data_frame.iloc[idx, 0])
                src_image_path = path.join(
                    org_dataset_path, 'images', image_path)
                dst_image_path = path.join(
                    dataset_path, str(fold_idx), "train", image_path)
                copy(src_image_path, dst_image_path)
            data_frame.loc[train_index].to_csv(path.join(
                dataset_path, str(fold_idx), "train.csv"), index=False)

            # Copy val csv and images to fold
            for idx in val_index:
                # image_path = str(data_frame.iloc[idx, 0]) + '.jpg'
                image_path = str(data_frame.iloc[idx, 0])
                src_image_path = path.join(
                    org_dataset_path, 'images', image_path)
                dst_image_path = path.join(
                    dataset_path, str(fold_idx), "val", image_path)
                copy(src_image_path, dst_image_path)
            data_frame.loc[val_index].to_csv(path.join(
                dataset_path, str(fold_idx), "val.csv"), index=False)

            fold_idx += 1


# =================================================================================
#                        Cornell Birdcall Identification
# =================================================================================

def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def scale_minmax(X, min=0.0, max=1.0, eps=1e-6):
    X_std = (X - X.min()) / (X.max() - X.min() + eps)
    X_scaled = X_std * (max - min) + min
    return X_scaled


def add_noise(audio, coeff):
    """Adds noise to provided audio clip

    Args:
        audio: audio file wihtout noise
        coeff: coeff is the proportion of noise to be added
    """

    noise_dir = Path(DATA_ROOT_DIR) / "bird_song" / "noise"

    noise_file_path = random.choice(listdir(noise_dir))
    noise_file_path = noise_dir / noise_file_path

    n_y, n_sr = sf.read(noise_file_path)
    length = n_y.shape[0]
    effective_length = n_sr * 5

    if length < effective_length:
        new_y = np.zeros(effective_length, dtype=n_y.dtype)
        start = np.random.randint(effective_length - length)
        new_y[start:start + length] = n_y
        n_y = new_y.astype(np.float32)
    elif length > effective_length:
        start = np.random.randint(length - effective_length)
        n_y = n_y[start:start + effective_length].astype(np.float32)
    else:
        n_y = n_y.astype(np.float32)

    noise_energy = np.sqrt(n_y.dot(n_y))
    audio_energy = np.sqrt(audio.dot(audio))

    audio += coeff * n_y * (audio_energy / noise_energy)
    return audio, n_sr

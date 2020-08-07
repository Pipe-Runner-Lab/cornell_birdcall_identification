import torch
import cv2
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from skimage import io
from pathlib import Path

from torch.utils.data import Dataset
from utils.noise import add_noise
from Datasets.utils import scale_minmax
from utils.submission_utils import BIRD_CODE

NUMBER_OF_FOLDS = 5
DATASET_NAME = 'bird_song'
PERIOD = 5
MEL_PARAMS = {'n_mels': 128, 'fmin': 20, 'fmax': 16000}


class Bird_Song_Dataset(Dataset):
    def __init__(self, mode, data_path, transformer=None, fold_number=0, noise=False):
        if transformer is None:
            raise Exception("transformer missing!")
        if fold_number is None or fold_number >= NUMBER_OF_FOLDS:
            raise Exception("fold limit exceeded!")
        if mode == "TRA":
            self.mode = "train"
        elif mode == "VAL":
            self.mode = "val"
        else:
            raise Exception(
                "{} not available for {}".format(mode, DATASET_NAME))
        self.fold_number = fold_number
        self.transformer = transformer
        self.csv_path = data_path / ("train.csv")
        self.data_dir = data_path / "audio"
        self.noise = noise

        self.data_frame = pd.read_csv(self.csv_path)
        if self.mode == "TRA":
            self.data_frame = self.data_frame[
                self.data_frame["fold"] != self.fold_number
            ]
        if self.mode == "VAL":
            self.data_frame = self.data_frame[
                self.data_frame["fold"] == self.fold_number
            ]

    def get_csv_path(self):
        return self.csv_path

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get basic data path
        ebird_code = self.data_frame["ebird_code"][idx]
        filename = self.data_frame["filename"][idx]

        # generate file path
        audio_filepath = self.data_dir / ebird_code / filename

        y, sr = sf.read(audio_filepath)

        length = y.shape[0]
        effective_length = sr * 5

        if length < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - length)
            new_y[start:start + length] = y
            y = new_y.astype(np.float32)
        elif length > effective_length:
            start = np.random.randint(length - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        # Adding noise
        if self.noise and self.mode == "train":
            y, sr = add_noise(y, 0.7)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **MEL_PARAMS)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        image = scale_minmax(melspec, 0, 255).astype(np.uint8)
        image = 255 - image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # converting to one hotvector
        label = np.zeros(len(BIRD_CODE), dtype="f")
        label[BIRD_CODE[ebird_code]] = 1
        return self.transformer["image"](image), label

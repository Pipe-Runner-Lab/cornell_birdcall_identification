import torch
import cv2
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from skimage import io
from pathlib import Path
import random
import ast

from torch.utils.data import Dataset

from Datasets.utils import scale_minmax
from utils.submission_utils import BIRD_CODE
from WaveTransformers.utils import (trim_audio_data, get_audio_data)

NUMBER_OF_FOLDS = 5
DATASET_NAME = 'bird_song'
PERIOD = 5
MEL_PARAMS = {'n_mels': 128, 'fmin': 20, 'fmax': 16000}


class Bird_Song_Dataset(Dataset):
    def __init__(self, mode, data_path, transformer=None, fold_number=0, mel_spec=True, multi_label=False):
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

        self.csv_path = data_path / "train.csv"
        self.data_dir = data_path / "audio"

        self.mel_spec = mel_spec
        self.multi_label = multi_label

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

    def mix_audio_data(self, y, sr, mix_idx_list, ebird_codes, coeff=0.6, k=2):
        # select idx from list
        mix_idx_list = random.choices(ast.literal_eval(
            mix_idx_list), weights=(25, 25, 25, 25), k=k)

        for mix_idx in mix_idx_list:
            ebird_code = self.data_frame["ebird_code"][mix_idx]
            filename = self.data_frame["filename"][mix_idx]
            audio_filepath = self.data_dir / ebird_code / filename

            m_y, m_sr = get_audio_data(audio_filepath)

            noise_energy = np.sqrt(m_y.dot(m_y))
            audio_energy = np.sqrt(y.dot(y))

            y += coeff * m_y * (audio_energy / noise_energy)

            ebird_codes.append(ebird_code)

        return y, sr, ebird_codes

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get basic data path
        ebird_codes = [self.data_frame["ebird_code"][idx]]
        filename = self.data_frame["filename"][idx]

        # generate file path
        audio_filepath = self.data_dir / ebird_codes[0] / filename

        # original
        y, sr = get_audio_data(audio_filepath, max_length=5)

        # multi-label mix
        if self.multi_label and self.mode == "train":
            y, sr, ebird_codes = self.mix_audio_data(
                y, sr, self.data_frame.at[idx, "mix"], ebird_codes)

        # wave transformation
        if self.transformer["wave"]:
            y, sr = self.transformer["wave"](y, sr)

        # Check
        #librosa.output.write_wav('audio_test.wav', y, sr)

        # converting to one hotvector
        label = np.zeros(len(BIRD_CODE), dtype="f")
        for ebird_code in ebird_codes:
            label[BIRD_CODE[ebird_code]] = 1

        if self.mel_spec:
            melspec = librosa.feature.melspectrogram(y, sr=sr, **MEL_PARAMS)
            melspec = librosa.power_to_db(melspec).astype(np.float32)

            image = scale_minmax(melspec, 0, 255).astype(np.uint8)
            image = 255 - image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # spec Image transformation
            return self.transformer["image"](image), label
        else:
            y = torch.from_numpy(y)
            return y, label

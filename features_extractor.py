#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pathlib
import cv2
import numpy as np
import librosa
from tqdm.auto import tqdm
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from pyts.approximation import PiecewiseAggregateApproximation
import pandas as pd
import os

def normalize(data):
    xmax, xmin = data.max(), data.min()
    return 2 * ((data - xmin) / (xmax - xmin)) - 1

def audio_mfcc(signal, sr, n_mfcc=30):
    mfcc_signal = np.mean(librosa.feature.mfcc(y=signal, sr=sr, 
                                              n_mfcc=n_mfcc, 
                                              fmin=300., fmax=600.,
                                              center=True, n_mels=20, 
                                              n_fft=1024), axis=0)
    return normalize(mfcc_signal)

def approximate_ts(X, window_size):
    paa = PiecewiseAggregateApproximation(window_size=window_size)
    return paa.transform(X)

def timeSeriesToImage(ts, kind="GADF", window_size=0):
    if window_size != 0:
        ts = approximate_ts(ts.reshape(1, -1), window_size)
        ts = ts.reshape(-1,1)

    gasf = GramianAngularField(method='summation')
    gadf = GramianAngularField(method='difference')
    mtf = MarkovTransitionField()
    rp = RecurrencePlot()

    if kind == "GADF":
        img = gadf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "GASF":
        img = gasf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "MTF":
        img = mtf.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "RP":
        img = rp.fit_transform(pd.DataFrame(ts).T)[0]
    elif kind == "RGB_GAF":
        gasf_img = gasf.fit_transform(pd.DataFrame(ts).T)[0]
        gadf_img = gadf.fit_transform(pd.DataFrame(ts).T)[0]
        img = np.dstack((gasf_img, gadf_img, np.zeros(gadf_img.shape)))
    return img

def generate_spectrogram(aud, Fs):
    S = librosa.feature.melspectrogram(y=aud, sr=Fs, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_norm = 255 * (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    return S_norm.astype(np.uint8)

def extract_features(DATASET_FILE, n_mfcc, kind, res_sig_size, features_folder):
    saveTo = f"{features_folder}/{kind}_dataset_{n_mfcc}_{res_sig_size}/"

    files = sorted(list(pathlib.Path(DATASET_FILE).rglob("*.wav")))
    for f in tqdm(files, total=len(files)):
        label = f.parts[-2]
        pathlib.Path(os.path.join(saveTo, label)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(features_folder, "mel_spectrogram", label)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(features_folder, "mfcc", label)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(features_folder, "rms", label)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(features_folder, "zcr", label)).mkdir(parents=True, exist_ok=True)

        signal, sr = librosa.load(f, duration=5.0)

        zcr = librosa.feature.zero_crossing_rate(signal)
        S, phase = librosa.magphase(librosa.stft(signal))
        rms = librosa.feature.rms(S=S)

        mfcc_feature = audio_mfcc(signal, sr, n_mfcc=n_mfcc)

        x = len(mfcc_feature) // res_sig_size
        img = timeSeriesToImage(mfcc_feature, kind=kind, window_size=x)
        img_path = os.path.join(saveTo, label, f.stem + ".png")
        cv2.imwrite(img_path, img)

        spectrogram_img = generate_spectrogram(signal, sr)
        cv2.imwrite(os.path.join(features_folder, "mel_spectrogram", label, f.stem + ".png"), spectrogram_img)

        np.save(os.path.join(features_folder, "mfcc", label, f.stem + ".npy"), mfcc_feature)
        np.save(os.path.join(features_folder, "rms", label, f.stem + ".npy"), rms)
        np.save(os.path.join(features_folder, "zcr", label, f.stem + ".npy"), zcr)

import common

if __name__ == "__main__":
    extract_features(common.AUG_AUDIO_DATASET, 20, "GADF", 90, features_folder=common.FEATURES_FOLDER)

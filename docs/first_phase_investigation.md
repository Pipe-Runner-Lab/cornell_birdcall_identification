# Important Threads
1. [x] https://www.kaggle.com/c/birdsong-recognition/discussion/159123 (Competition expectation by host)
  - Main challenge is the shift in domain of data from training to inference ⭐️
  - Train a classfier to work well on clean audio --> Fine tune it to work well on noisy audio ⭐️
  - **Be aware that not all species mentioned in the secondary label column might be part of the training data.** 
  - For training, you can focus on primary and secondary labels. ⭐️
  - For a **submission**, you should use the **ebird code**.  ⭐️
  - You should definitely consider using secondary labels to validate your multi-class classifier, *but (again) be aware that these labels are only weak labels with no timestamp*. 🔪
  - "**secondary_labels will be better than background**. But both are almost same and week labels regarding label quality"

# Techniques
1. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/158943 (Previous competition first prize)
2. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/158908 (Audio preprocessing techniques)
3. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/159001 (Spectrogram conversion using Librosa)
4. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/160454 (Cocktail party speaker seperation)
5. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/161505 (Wavenet/Model suggestion)
6. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/164230 (MFCC)
7. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/161858 (Background noise as additional class)

# External Data
1. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/160293 (Xeno-canto)
2. [x] https://www.kaggle.com/c/birdsong-recognition/discussion/159943 (Resampled and Reorganized train data)

# Notebooks
  
## Implementation Guides
1. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/159993 (Deal with hidden data without submit button - Faster Audio file read)
2. [ ] https://www.kaggle.com/hidehisaarai1213/inference-pytorch-birdcall-resnet-baseline (Inference Pipeline)
3. [ ] https://www.kaggle.com/frlemarchand/bird-song-classification-using-an-lstm (LSTM for classificaiton)
4. [x] https://www.kaggle.com/ttahara/training-birdsong-baseline-resnest50-fast#Birdsong-Pytorch-Baseline:-ResNeSt50-fast-(Training) (Resnet50 training pytorch) ⭐️
5. [x] https://www.kaggle.com/ttahara/inference-birdsong-baseline-resnest50-fast (Resnet 50 infrence) ⭐️
6. [x] https://www.kaggle.com/c/birdsong-recognition/discussion/160222 (FastAI training / 0.48 LB)
  - https://github.com/earthspecies/birdcall ❓
  - https://www.kaggle.com/c/birdsong-recognition/discussion/161860 ❓

## EDA
1. [x] https://www.kaggle.com/pavansanagapati/birds-sounds-eda-spotify-urban-sound-eda
  - Max Sampling Rate of audio is 44100 Hx and 48000 Hz
  - Maximum rating of train audio files between 3.5-5.0  

2. [x] https://www.kaggle.com/andradaolteanu/birdcall-recognition-eda-and-audio-fe
  - Majority of the data was registered between 2013 and 2019, during Spring and Summer months
  - Pitch is usually unspecified. This is one of the more miscellaneous columns, that we need to be careful how we interpret. Most Song Types are call, song or flight.
  - In most recordings the birds were seen, usually at an altitude between 0m and 10m.
  - The majority of recordings are located in the US, followed by Canada and Mexico.
  - train_audio: short recording (majority in mp3 format) of INDIVIDUAL birds.
  - test_audio: recordings took in 3 locations:
  - Site 1 and Site 2: recordings 10 mins long (mp3) that have labeled a bird every 5 seconds. This is meant to mimic the real life scenario, when you would usually have more than 1 bird (or no bird) singing.
  - Site 3: recordings labeled at file level (because it is especially hard to have coders trained to label these kind of files)
  - Maximum recordings in train have length <= 100sec

3. [x] https://www.kaggle.com/hamditarek/audio-data-analysis-using-librosa/notebook#Librosa
  - Discussion about librosa library: Librosa returns times series ndarray of audio with default sampling rate = 22KHz Mono
  - Description about Spectrogram, spectral rolloff, MFCC     https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d
    - Audio is 3D signal with time, amplitude and frequency as axis
    - Waveplot shows loudness of music in amplitude form (amp vs time)
    - Spectrogram is visual representation of spectrum of frequencies as they vary with time
    - Zero Crossing Rate is the rate of change of signal from +ve to -ve and vice versa
    - Spectral Centroid shows the centre of mass for a sound based on frequency. It is calculated as the weighted mean of the frequencies  present in the signal, determined using a Fourier transform
    - Spectral Rolloff is the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies. It also gives results for each frame.
    - MFCC — Mel-Frequency Cepstral Coefficients are a small set of features (usually about 10–20) which concisely describe the overall shape of a spectral envelope.

***

# Prior work (Discussion Threads and Notebooks)
1. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/158933
2. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/161787 (Paper list)
3. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/160759 (Free Sound challenge)

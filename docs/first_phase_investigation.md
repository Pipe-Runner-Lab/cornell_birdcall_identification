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
2. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/159943 (Resampled and Reorganized train data)

# Notebooks
  
## Implementation Guides
1. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/159993 (Deal with hidden data without submit button - Faster Audio file read)
2. [ ] https://www.kaggle.com/hidehisaarai1213/inference-pytorch-birdcall-resnet-baseline (Inference Pipeline)
3. [ ] https://www.kaggle.com/frlemarchand/bird-song-classification-using-an-lstm (LSTM for classificaiton)
4. [ ] https://www.kaggle.com/ttahara/training-birdsong-baseline-resnest50-fast#Birdsong-Pytorch-Baseline:-ResNeSt50-fast-(Training) (Resnet50 training pytorch)
5. [ ] https://www.kaggle.com/ttahara/inference-birdsong-baseline-resnest50-fast (Resnet 50 infrence)
6. [x] https://www.kaggle.com/c/birdsong-recognition/discussion/160222 (FastAI training / 0.48 LB)
  - https://github.com/earthspecies/birdcall ❓
  - https://www.kaggle.com/c/birdsong-recognition/discussion/161860 ❓

## EDA
1. [ ] https://www.kaggle.com/pavansanagapati/birds-sounds-eda-spotify-urban-sound-eda
2. [ ] https://www.kaggle.com/andradaolteanu/birdcall-recognition-eda-and-audio-fe
3. [ ] https://www.kaggle.com/hamditarek/audio-data-analysis-using-librosa/notebook#Librosa

***

# Prior work (Discussion Threads and Notebooks)
1. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/158933
2. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/161787 (Paper list)
3. [ ] https://www.kaggle.com/c/birdsong-recognition/discussion/160759 (Free Sound challenge)

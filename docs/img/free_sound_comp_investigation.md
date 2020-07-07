Link: https://github.com/lRomul/argus-freesound

## Basic

#### Preprocession
- Converted to spectrogram for applying CNN

#### Augmentations
- specAugment
```
size = 256
transforms = Compose([
    OneOf([
        PadToSize(size, mode='wrap'),      # Reapeat small clips
        PadToSize(size, mode='constant'),  # Pad with a minimum value
    ], p=[0.5, 0.5]),
    RandomCrop(size),                      # Crop 256 values on time axis 
    UseWithProb(
        # Random resize crop helps a lot, but I can't explain why ¯\_(ツ)_/¯   
        RandomResizedCrop(scale=(0.8, 1.0), ratio=(1.7, 2.3)),
        prob=0.33
    ),
    # SpecAugment [1], masking blocks of frequency channels, and masking blocks of time steps
    UseWithProb(SpecAugment(num_mask=2,       
                            freq_masking=0.15,
                            time_masking=0.20), 0.5),
    # Use librosa.feature.delta with order 1 and 2 for creating 2 additional channels 
    # then divide by 100 
    ImageToTensor()                  
])
```

- Mixup (Cretes new sample based on the weighted average of two samples)
  - Author also used SigmoidConcatMixer

#### Model
A modified version of https://www.kaggle.com/mhiro2/simple-2d-cnn-classifier-with-pytorch was used; the modifications include:
  - attention
  - skip connection
  - auxilary classifiers

#### Training
- 5 random folds
- Loss: BCE on curated, Lsoft with beta 0.7 on noisy data ❓
- Optimizer: Adam with initial LR 0.0009
- LR scheduler: Reduce on a plateau with patience 6, factor 0.6
- Use different probabilities for sampling curated and noisy data ❓
- Training on hand relabeled curated samples with a low lwlrap score by previous models ❓
- Training with BCE on noisy samples with a high **lwlrap** score by previous models ❓
- Mixed precision training with apex.amp allows using batch size 128 with input size 256x128 px (Mixed precision training) ❓

#### Ensemble
The geometric mean of 7 first-level models and 3 second-level models was used for the final submission. 
MLPs trained with different hyperparameters were used as second-level models. 
Seven first-level models were chosen by enumeration of combinations of training experiments to finding the highest CV score.

## Suplimentry 
1. [x] https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data (Spectrogram convertion)- 
 - sampling_rate = 44100
 - hop_length = 345 * 2
 - fmin = 20
 - fmax = sampling_rate // 2
 - n_mels = 128
 - n_fft = n_mels * 20
 - min_seconds = 0.5
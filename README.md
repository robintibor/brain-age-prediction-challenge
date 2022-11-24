# Code for Submission to Brain Age Prediction Challenge

This is the code for team robintibor.

## Approach
We train a deep4 network  from [braindecode](www.braindecode.org) using cropped training.

## Model
We double the number of the default number of channels and add a sigmoid function at the end (targets are min-max scaled for training purposes).

### Preprocessing

As for preprocessing, offline we:
1. resample the data to 100 Hz
2. remove Cz sensor

Online, we do this on each input window:
1. FFT-based highpass above 0.5 Hz
2. robustly (median-based) standardize each channel 
3. clip values
4. common average rereference the data
5. clip values again

An input window is 15 seconds long. For standardization, we first subtract the median value from each channel and then divide by the median absolute value, again for each channel separately.

### Model Training
We train with AdamW with learning rate `3e-4` and weight_decay `1e-4`, randomly dropout eeg electrode channels with `p=0.6` (probability of each channel being dropped) for 30 epochs with batch size 64. We use 10 additional epochs of stochastic weight averaging at the end and use the averaged weights to make our final predictions. We also build an ensemble from models trained with different seeds, we use the 3 models with the lowest train error for our final submission.
# Code for Submission to Brain Age Prediction Challenge

This is the code for team robintibor.

## Approach
We train a deep4 network from [braindecode](www.braindecode.org), with double the number of the default number of channels.

### Preprocessing
As for preprocessing,:
1. resample the data to 100 Hz
2. remove Cz sensor
3. robustly (median-based) standardize each channel in each input window
4. clip values
5. common average rereference the data
6. clip values again

For the standardization, we separate each input window for the network separately, an input window is 15 seconds long. We first subtract the median value from each channel and then divide by the median absolute value, for each channel separately.

### Model Training
We train with AdamW, randomly dropout eeg electrode channels with `p=0.6` (probability of each channel being dropped). We use stochastic weight averaging at the end to get better performance.
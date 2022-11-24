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


### Code
First, you need to run `PrepareData.ipynb` with adjusted paths for your machine. Then `run_exp` function in `run_exp.py` needs to be called. Finally, `EnsemblePredictions.ipynb` shows how to build the ensemble.

This was the json config used for the final submission (arguments for `run_exp` function in `run_exp.py`)

```
{
  "batch_size": 64,
  "channel_drop_p": 0.6,
  "clip_val_after_car": 4,
  "clip_val_before_car": 6,
  "common_average_rereference": true,
  "debug": false,
  "drop_out_p": 0,
  "eval_mode": "eval",
  "final_nonlin": "sigmoid",
  "first_n": null,
  "low_cut_hz": 0.5,
  "lr": 0.0003,
  "lr_schedule": null,
  "merge_restart_models": false,
  "n_epochs": 30,
  "n_restart_epochs": null,
  "n_start_filters": 50,
  "n_swavg_epochs": 10,
  "norm_layer": "layernorm",
  "np_th_seed": 4, # 4,2 and 8 were used for final ensemble
  "optim_wrapper": null,
  "save_valid_preds": true,
  "train_on_valid": true,
  "training_mode": "train",
  "weight_decay": 0.0001
}
``` 
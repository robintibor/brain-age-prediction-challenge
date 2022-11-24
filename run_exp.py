import math
import json
import glob
import pickle
import logging
import warnings
import argparse
from datetime import datetime
from functools import partial
from collections import OrderedDict
from io import StringIO as StringBuffer

import mne
mne.set_log_level('ERROR')
#mne.set_config("MNE_LOGGING_LEVEL", "ERROR")
import torch
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_color_codes('deep')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.set_loglevel('ERROR')
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler, Checkpoint, TrainEndCheckpoint, ProgressBar, BatchScoring
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from skorch.utils import valid_loss_score, noop

from braindecode.datasets import BaseDataset, BaseConcatDataset
from braindecode.datasets.tuh import TUHAbnormal
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.preprocessing.windowers import create_fixed_length_windows
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, to_dense_prediction_model, Deep4Net, TCN
from braindecode.models.modules import Expression
from braindecode.regressor import EEGRegressor
from braindecode.classifier import EEGClassifier
from braindecode.training import CroppedLoss, CroppedTrialEpochScoring
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s : %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger()
logger.setLevel("DEBUG")

from braindecode.datasets import BaseConcatDataset
import os
from rtsutils.nb_util import Results
from tqdm.autonotebook import trange
from tqdm.autonotebook import tqdm
from decode_tueg import create_windows
from decode_tueg import get_n_preds_per_input
from decode_tueg import get_model
from decode_tueg import avg_swa_fn
import torch as th
from decode_tueg import soft_clamp_to_0_1
from sam import SAM
from sam import enable_running_stats, disable_running_stats
from torch import nn
from braindecode.augmentation import ChannelsDropout
import pickle
from decode_tueg import RobustStandardizeBatch, ClipBatch, CommonAverageReference
from decode_tueg import Add
from braindecode.training.scoring import trial_preds_from_window_preds
from braindecode.util import set_random_seeds
from torch.optim.swa_utils import AveragedModel, SWALR
from decode_tueg import HighPassFFT


def run_exp(
        first_n,
        clip_val_before_car,
        clip_val_after_car,
        train_on_valid,
        n_epochs,
        batch_size,
        final_nonlin,
        drop_out_p,
        norm_layer,
        lr_schedule,
        optim_wrapper,
        lr,
        weight_decay,
        n_swavg_epochs,
        channel_drop_p,
        training_mode,
        eval_mode,
        np_th_seed,
        output_dir,
        n_restart_epochs,
        merge_restart_models,
        n_start_filters,
        low_cut_hz):   
    trange = range
    tqdm = lambda x: x
    set_random_seeds(np_th_seed, cuda=True)
    # cosine annealing, weight averaging, SAM optimizer
    assert norm_layer in ["layernorm", "batchnorm", None]
    assert training_mode in ['train', 'eval']
    assert eval_mode in ['train', 'eval']
    assert not (merge_restart_models and (n_restart_epochs is None))
    if optim_wrapper == 'sam':
        sam_rho = 0.05
    elif optim_wrapper == 'asam':
        sam_rho = 0.5
    data_path = '/work/dlclarge2/schirrmr-eeg-age-competition/pkl-datasets/'
    tuabn_train = pickle.load(open(os.path.join(data_path, 'tuabn_train.pkl'), 'rb'))
    tuabn_valid = pickle.load(open(os.path.join(data_path, 'tuabn_valid.pkl'), 'rb'))

    ch_names = tuabn_train.datasets[0].windows.ch_names
    # needed later for inverting target transform
    target_scaler = tuabn_train.target_transform
    if first_n is not None:
        tuabn_train = th.utils.data.Subset(tuabn_train, list(range(first_n)))
        tuabn_valid = th.utils.data.Subset(tuabn_valid, list(range(first_n)))
    
    if train_on_valid:
        tuabn_train = BaseConcatDataset(
            (tuabn_train, tuabn_valid),
            target_transform=target_scaler)

    train_loader = th.utils.data.DataLoader(
        tuabn_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2)

    valid_loader = th.utils.data.DataLoader(
        tuabn_valid,
        batch_size=128,
        shuffle=False,
        num_workers=2)

    target_name = 'age'
    model_name = 'deep'
    cropped = True
    n_channels = len(ch_names)
    window_size_samples = 1500
    squash_outs = 1
    cuda = True

    model, _, _ = get_model(
        n_channels,
        np_th_seed,
        cuda,
        target_name,
        model_name,
        cropped,
        window_size_samples,
        squash_outs,
        n_start_filters=n_start_filters,
    )

    preproc_modules = [
        RobustStandardizeBatch(1e-10),
        ClipBatch(-clip_val_before_car, clip_val_before_car),
        CommonAverageReference(), 
        ClipBatch(-clip_val_after_car, clip_val_after_car),]

    if low_cut_hz is not None:
        preproc_modules = [HighPassFFT(low_cut_hz=low_cut_hz)] + preproc_modules

    model = nn.Sequential(
        *preproc_modules,
        model,);
    model = model.cuda();

    deep_model = model[-1].deep

    for m in deep_model:
        if m.__class__.__name__ == 'Dropout':
            m.p = drop_out_p


    X,y,i = next(train_loader.__iter__())
    X = X.cuda()

    cur_X = X
    wanted_modules = []
    for m in deep_model:
        if (m.__class__.__name__ != 'BatchNorm2d') or (norm_layer == 'batchnorm'):
            wanted_module = m
        else:
            if norm_layer == "layernorm":
                wanted_module = nn.LayerNorm(cur_X.shape[1:], elementwise_affine=True).cuda()
            else:
                assert norm_layer is None
                continue
        cur_X = wanted_module(cur_X)
        wanted_modules.append(wanted_module)

    #recheck that everything works
    cur_X = X
    for wanted_module in wanted_modules:
        cur_X = wanted_module(cur_X)

    model[-1].deep = nn.Sequential(*wanted_modules)


    if final_nonlin == 'sigmoid':
        model[-1].sigmoid = nn.Sigmoid()
    elif final_nonlin == 'soft_clamp_0_1':
        model[-1].sigmoid = nn.Sequential(
            Add(),
            Expression(soft_clamp_to_0_1))
    else:
        assert final_nonlin is None
        model[-1].sigmoid = Add()

    opt_model = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_for_scheduler = opt_model

    if optim_wrapper == 'sam' or optim_wrapper == 'asam':
        opt_model = SAM(model.parameters(), th.optim.AdamW, rho=sam_rho, adaptive=optim_wrapper == 'asam',
                       lr=lr, weight_decay=weight_decay)

        opt_for_scheduler = opt_model.base_optimizer

    if lr_schedule == 'cosine' and n_restart_epochs is None:
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(opt_for_scheduler, T_max=len(train_loader) * n_epochs)
    elif lr_schedule == 'cosine' and n_restart_epochs is not None:
        scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_for_scheduler, T_0=len(train_loader) * n_restart_epochs)
    else:
        assert lr_schedule == None
        # identity scheduler, no change of lr, lr will always be multiplied with 1
        scheduler = th.optim.lr_scheduler.LambdaLR(opt_for_scheduler, lambda i_step: 1)

    channel_dropout = ChannelsDropout(1, channel_drop_p,)
    state_dicts = []
    nb_res = Results()
    for i_epoch in trange(n_epochs):
        for X,y,i in tqdm(train_loader):
            model.train(training_mode == 'train')
            X = X.cuda()
            y = y.cuda()

            X = channel_dropout(X)
            out = model(X)
            loss = th.mean(th.abs(y-out.mean(axis=(1,2))))
            opt_model.zero_grad(set_to_none=True)
            loss.backward()
            if optim_wrapper == None:
                opt_model.step()
            else:
                assert optim_wrapper == 'sam' or optim_wrapper == 'asam'
                opt_model.first_step(zero_grad=True)
                out = model(X)
                disable_running_stats(model)
                loss = th.mean(th.abs(y-out.mean(axis=(1,2))))
                loss.backward()
                opt_model.second_step(zero_grad=True)
                enable_running_stats(model)

            scheduler.step()
            opt_model.zero_grad(set_to_none=True)
            nb_res.collect(loss=loss.item())
            nb_res.print()

        if n_restart_epochs is not None and ((i_epoch % n_restart_epochs) == 1) and merge_restart_models:
            state_dicts.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        model.train(eval_mode == 'train')
        all_losses = []
        for X,y,i  in tqdm(valid_loader):
            X = X.cuda()
            y = y.cuda()
            with th.no_grad():
                outs = model(X)
                loss = th.abs(y-outs.mean(axis=(1,2)))
                all_losses.append(loss.detach().cpu().numpy())
        print(np.mean(all_losses))

    if merge_restart_models:
        merged_state_dict = {}
        for key in state_dicts[0].keys():
            vals = [s[key] for s in state_dicts]
            with th.no_grad():
                merged_state_dict[key] = th.stack(vals,dim=0).mean(dim=0).cuda()
        model.load_state_dict(merged_state_dict)

    averaged_model = AveragedModel(model, avg_fn=avg_swa_fn)
    # training loop
    if n_swavg_epochs > 0:
        if lr_schedule == 'cosine':
            # reset in case was reduced by cosine annealing?
            opt_for_scheduler.param_groups[0]['lr']  = lr / 10
        for _ in trange(n_swavg_epochs):
            for X,y,i in tqdm(train_loader):
                X = X.cuda()
                y = y.cuda()
                model.train(training_mode == 'train')
                out = model(X)
                loss = th.mean(th.abs(y-out.mean(axis=(1,2))))
                opt_for_scheduler.zero_grad(set_to_none=True)
                loss.backward()
                opt_for_scheduler.step()
                opt_for_scheduler.zero_grad(set_to_none=True)
            averaged_model.update_parameters(model)

    model = averaged_model.module
        
        
    model.train(eval_mode == 'train')

    all_losses = []
    for X,y,i  in tqdm(train_loader):
        X = X.cuda()
        y = y.cuda()
        with th.no_grad():
            outs = model(X)
            loss = th.abs(y-outs.mean(axis=(1,2)))
            all_losses.append(loss.detach().cpu().numpy())

    train_loss = np.mean(all_losses)
    print(np.mean(all_losses))
    
    
    all_y = []
    all_outs = []
    all_losses = []
    all_is = []
    for X,y,i  in tqdm(valid_loader):
        X = X.cuda()
        y = y.cuda()
        with th.no_grad():
            outs = model(X)
            loss = th.abs(y-outs.mean(axis=(1,2)))
            all_losses.append(loss.detach().cpu().numpy())
        all_outs.append(outs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
        all_is.append(i)
    print(np.mean(all_losses))
    valid_loss = np.mean(all_losses)

    if first_n is None:
        i_window_in_trials = np.concatenate(
            [i[0].cpu().numpy() for i in all_is])
        i_window_stops = np.concatenate(
            [i[2].cpu().numpy() for i in all_is]
        )
        all_outs_flat = [o for out in all_outs for o in out]
        trial_preds = trial_preds_from_window_preds(
                    all_outs_flat,
                    i_window_in_trials,
                    i_window_stops)
        window_0_per_trial_mask = np.diff(i_window_in_trials, prepend=[np.inf]) != 1
        trial_ys = np.concatenate(all_y)[window_0_per_trial_mask]
        mean_trial_preds = np.array([p.mean() for p in trial_preds])
        assert np.allclose(trial_ys[tuabn_valid.description.condition == 'EC'],
                   trial_ys[tuabn_valid.description.condition == 'EO'])
        mean_pred_across_cond = (mean_trial_preds[tuabn_valid.description.condition == 'EC'] + 
         mean_trial_preds[tuabn_valid.description.condition == 'EO']) / 2
        valid_loss_trial = np.abs(trial_ys[tuabn_valid.description.condition == 'EC'] - mean_pred_across_cond).mean()
        valid_loss_trial_orig = (target_scaler.mult_element - target_scaler.add_element) * (valid_loss_trial)
    
    
    tuabn_eval = pickle.load(open(os.path.join(data_path,'tuabn_eval.pkl'), 'rb'))
    assert np.all(np.array([dset.windows.ch_names.index('Cz') for dset in tuabn_eval.datasets]) == 128)
    
    test_loader = th.utils.data.DataLoader(
        tuabn_eval,
        batch_size=128,
        shuffle=False,
       num_workers=2)
    all_outs = []
    all_is = []
    for X,y,i  in tqdm(test_loader):
        X = X.cuda()
        with th.no_grad():
            outs = model(X[:,:-1])
        all_outs.append(outs.detach().cpu().numpy())
        all_is.append(i)

    i_window_in_trials = np.concatenate(
        [i[0].cpu().numpy() for i in all_is])

    i_window_stops = np.concatenate(
        [i[2].cpu().numpy() for i in all_is]
    )

    all_outs_flat = [o for out in all_outs for o in out]
    trial_preds = trial_preds_from_window_preds(
                all_outs_flat,
                i_window_in_trials,
                i_window_stops)

    mean_trial_preds = np.array([p.mean() for p in trial_preds])
    inverted_trial_preds = target_scaler.invert(mean_trial_preds)
    subject_trial_preds = (mean_trial_preds[:400] + mean_trial_preds[400:]) / 2
    inverted_preds = target_scaler.invert(subject_trial_preds)

    tuabn_eval_df = pd.DataFrame({'id': tuabn_eval.description.subject.iloc[:400], 'age': inverted_preds})
    tuabn_eval_df.to_csv(os.path.join(
        output_dir, './submission.csv'), index=False)
    tuabn_eval_df_both_conditions = pd.DataFrame({'id': tuabn_eval.description.subject, 'age': inverted_trial_preds})
    tuabn_eval_df_both_conditions.to_csv(os.path.join(
        output_dir, './submission_both_cond.csv'), index=False)
    
    th.save(model, os.path.join(output_dir, 'model.th'))
    
    
    inverted_valid_targets = target_scaler.invert(np.array(all_y).reshape(-1))
    fig = plt.figure(figsize=(8,3))
    plt.plot(np.linspace(0,100,len(inverted_preds)), sorted(inverted_preds), label="Test Predictions")
    plt.plot(np.linspace(0,100,len(inverted_valid_targets)), sorted(inverted_valid_targets), label="Valid Targets")
    plt.title("Cumulative Distribution Function Valid Targets And Test Predictions")
    plt.legend(bbox_to_anchor=(1,1,0,0))
    plt.xlabel("Fraction of Datapoints Below That Age [%]")
    plt.ylabel("Age [years]")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "CDFfig.png"), dpi=300, bbox_inches='tight')
    
    results = {
        'train_loss': train_loss,
        'train_loss_orig': train_loss * (target_scaler.mult_element- target_scaler.add_element),
        'valid_loss': valid_loss,
        'valid_loss_orig': valid_loss * (target_scaler.mult_element- target_scaler.add_element),}
    
    if first_n is None:
        results['valid_loss_trial'] = valid_loss_trial
        resultſ['valid_loss_trial_orig'] = valid_loss_trial_orig
    return results
               
               
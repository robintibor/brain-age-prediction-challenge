import os

os.sys.path.insert(0, '/home/schirrmr/code/utils/')
os.sys.path.insert(0, '/home/schirrmr/braindecode/code/braindecode/tmpnoteb_hidefromalex/age-competition')
import time
import logging

from hyperoptim.parse import (
    cartesian_dict_of_lists_product,
    product_of_list_of_lists_of_dicts,
)

import torch as th
import torch.backends.cudnn as cudnn

logging.basicConfig(format="%(asctime)s | %(levelname)s : %(message)s")


log = logging.getLogger(__name__)
log.setLevel("INFO")


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product

    save_params = [
        {
            "save_folder": "/work/dlclarge2/schirrmr-eeg-age-competition/exps/",
        }
    ]

    debug_params = [
        {
            "debug": False,
        }
    ]

    data_params = dictlistprod(
        {
            "first_n": [None],
            "clip_val_before_car": [6],
            "clip_val_after_car": [4],
            "train_on_valid": [False,],#True
            "low_cut_hz": [1],
            "common_average_rereference": [False,],#True
        }
    )

    train_params = dictlistprod(
        {
            "n_epochs": [30,],
            "batch_size": [64],
            "final_nonlin": ['sigmoid'],#'soft_clip_0_1',None,
            "drop_out_p": [0],
            "norm_layer": ['layernorm'],  # True
            "lr_schedule": [None],  # False#"cosine", "cosine", 
            "optim_wrapper": [None],  # 4#'sam', 'asam'
            "lr": [3e-4],
            "weight_decay": [1e-4],#1e-41e-3,1e-2
            "n_swavg_epochs": [10],#5,],#0.05,10
            "channel_drop_p": [0.6,],
            "n_restart_epochs": [None],
            "merge_restart_models": [False,],
            "n_start_filters": [50],
        }
    )

    mode_params = dictlistprod(
        {
            "training_mode": ['train', ],
            "eval_mode": ['eval'],
        }) + dictlistprod(
        {
            #"training_mode": ['eval'],
            #"eval_mode": ['eval'],
        }
    )

    random_params = dictlistprod(
        {
            "np_th_seed": range(10),
        }
    )

    grid_params = product_of_list_of_lists_of_dicts(
        [
            save_params,
            debug_params,
            data_params,
            train_params,
            mode_params,
            random_params,
        ]
    )

    return grid_params


def sample_config_params(rng, params):
    return params


def run(
    ex,
    debug,
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
    n_restart_epochs,
    merge_restart_models,
    n_start_filters,
    low_cut_hz,
    common_average_rereference,
):
    if debug:
        n_epochs = 3
        first_n = 128
    kwargs = locals()
    kwargs.pop("ex")
    kwargs.pop("debug")
    if not debug:
        log.setLevel("INFO")
    file_obs = ex.observers[0]
    output_dir = file_obs.dir
    kwargs["output_dir"] = output_dir
    th.backends.cudnn.benchmark = True
    import sys

    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )
    start_time = time.time()
    ex.info["finished"] = False

    import os

    from run_exp import run_exp

    results = run_exp(**kwargs)
    end_time = time.time()
    run_time = end_time - start_time
    ex.info["finished"] = True

    for key, val in results.items():
        ex.info[key] = float(val)
    ex.info["runtime"] = run_time

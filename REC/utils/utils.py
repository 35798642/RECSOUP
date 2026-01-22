# -*- coding: UTF-8 -*-
"""
This file contains code adapted from a prior open-source implementation.

The original source and license will be properly acknowledged in the
camera-ready version.
"""
import os
import random
import logging
import torch
import datetime
import time
import numpy as np
import pandas as pd
from typing import List, Dict, NoReturn, Any, Union


# ==============================
# Random Seed Initialization
# ==============================
def init_seed(seed: int) -> None:
    """
    Initialize random seeds for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ==============================
# DataFrame Utilities
# ==============================
def df_to_dict(df: pd.DataFrame) -> dict:
    """
    Convert a DataFrame to a dictionary of numpy arrays.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Dictionary where each column is a numpy array.
    """
    res = df.to_dict('list')
    for key in res:
        res[key] = np.array(res[key])
    return res


def eval_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate string representations of lists in DataFrame columns.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with evaluated list columns.
    """
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(lambda x: eval(str(x)))
    return df


# ==============================
# GPU Utilities
# ==============================
def batch_to_gpu(batch: dict, device: torch.device) -> dict:
    """
    Move all tensors in a batch to a specified device.

    Args:
        batch (dict): Dictionary of tensors.
        device (torch.device): Target device.

    Returns:
        dict: Batch with tensors moved to device.
    """
    for c in batch:
        if isinstance(batch[c], torch.Tensor):
            batch[c] = batch[c].to(device)
    return batch


# ==============================
# Logging & Debugging
# ==============================
def check(check_list: List[tuple]) -> NoReturn:
    """
    Log selected tensors for inspection.

    Args:
        check_list (List[tuple]): List of (name, tensor) tuples.
    """
    logging.info('')
    for i, (name, tensor) in enumerate(check_list):
        d = tensor.detach().cpu().numpy()
        logging.info(os.linesep.join([
            f"{name}\t{d.shape}",
            np.array2string(d, threshold=20)
        ]) + os.linesep)


def format_metric(result_dict: Dict[str, Any]) -> str:
    """
    Format evaluation metrics as a readable string.

    Args:
        result_dict (Dict[str, Any]): Dictionary of metric results.

    Returns:
        str: Formatted string of metrics.
    """
    assert isinstance(result_dict, dict)
    format_str = []
    metrics = np.unique([k.split('@')[0] for k in result_dict.keys()])
    topks = np.unique([int(k.split('@')[1]) for k in result_dict.keys() if '@' in k])
    if not len(topks):
        topks = ['All']

    for topk in np.sort(topks):
        for metric in np.sort(metrics):
            name = f"{metric}@{topk}"
            m = result_dict[name] if topk != 'All' else result_dict[metric]
            if isinstance(m, (float, np.floating)):
                format_str.append(f"{name}:{m:.4f}")
            elif isinstance(m, (int, np.integer)):
                format_str.append(f"{name}:{m}")
    return ','.join(format_str)


def format_arg_str(args, exclude_lst: list, max_len: int = 20) -> str:
    """
    Format command-line arguments as a readable table string.

    Args:
        args: argparse.Namespace object.
        exclude_lst (list): List of argument names to exclude.
        max_len (int): Maximum value string length.

    Returns:
        str: Formatted argument table.
    """
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]

    key_max_len = max(len(k) for k in keys + ['Arguments'])
    value_max_len = max(min(max(len(str(v)) for v in values), max_len), len('Values'))
    horizon_len = key_max_len + value_max_len + 5

    res_str = linesep + '=' * horizon_len + linesep
    res_str += f" Arguments{' ' * (key_max_len - 9)} | Values{' ' * (value_max_len - 6)} {linesep}"
    res_str += '=' * horizon_len + linesep

    for key in sorted(keys):
        value = str(arg_dict[key]).replace('\t', '\\t')
        if len(value) > max_len:
            value = value[:max_len-3] + '...'
        res_str += f" {key}{' ' * (key_max_len - len(key))} | {value}{' ' * (value_max_len - len(value))}{linesep}"
    res_str += '=' * horizon_len
    return res_str


def check_dir(file_name: str) -> None:
    """
    Ensure the directory for a file exists; create if missing.

    Args:
        file_name (str): Path to file.
    """
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        print(f"Creating directories: {dir_path}")
        os.makedirs(dir_path)


def non_increasing(lst: list) -> bool:
    """
    Check if a list is non-increasing.

    Args:
        lst (list): Input list.

    Returns:
        bool: True if non-increasing.
    """
    return all(x >= y for x, y in zip([lst[0]]*(len(lst)-1), lst[1:]))


def get_time(day: bool = False) -> str:
    """
    Get current time as formatted string.

    Args:
        day (bool): If True, return only month-day.

    Returns:
        str: Formatted time string.
    """
    if day:
        return datetime.datetime.now().strftime("%m-%d")
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==============================
# Timer Context Manager
# ==============================
class timeit_context:
    """
    Context manager for measuring elapsed time.

    Usage:
        with timeit_context("Processing..."):
            ...
    """
    def __init__(self, msg: str = None, loglevel: int = logging.INFO) -> None:
        self.msg = msg
        self.loglevel = loglevel

    def _log(self, msg: str) -> None:
        logging.log(self.loglevel, msg)

    def __enter__(self) -> None:
        self.start_time = time.time()
        if self.msg:
            self._log(self.msg)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed_time = time.time() - self.start_time
        self._log(f"Elapsed time: {elapsed_time:.2f}s")


# ==============================
# Misc Utilities
# ==============================
def human_readable(num: Union[int, float]) -> str:
    """
    Convert a large number to a human-readable string with K/M/B suffixes.

    Args:
        num: Input number.

    Returns:
        str: Human-readable string.
    """
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return f"{num:.2f}{['','K','M','B','T','P'][magnitude]}"


def print_trainable_parameters(model: torch.nn.Module, verbose: bool = False) -> None:
    """
    Print number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): Model object.
        verbose (bool): If True, print each parameter shape.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            if verbose:
                print(f"{name}: {param.numel()}")
            trainable_params += param.numel()

    print(
        f"trainable params: {human_readable(trainable_params)} || "
        f"all params: {human_readable(all_param)} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

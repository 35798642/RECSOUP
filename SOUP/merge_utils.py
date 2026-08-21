
import time
import copy
import torch
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import logging


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )

def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    for key in remove_keys:
        if key in state_dict:
            sorted_reference_dict[key] = state_dict[key]

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict[
                "transformer.shared.weight"
            ]
    return sorted_reference_dict

def add_ptm_to_tv(tv_dict, ptm_dict):
    """
    Adds the values from ptm_dict to the corresponding values in tv_dict.

    This function ensures that both dictionaries have the same keys and then 
    adds the values from ptm_dict to the values in tv_dict for each key.

    Args:
        tv_dict (dict): A dictionary containing the initial values.
        ptm_dict (dict): A dictionary containing the values to be added to tv_dict.

    Returns:
        dict: A new dictionary with the same keys as tv_dict and ptm_dict, where 
              each value is the sum of the corresponding values from tv_dict and ptm_dict.

    Raises:
        AssertionError: If the keys of tv_dict and ptm_dict do not match.
    """
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict

def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )
            else:
                print("Parameter names match.")

def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True

def greater_than_std_mask(tensor, factor, return_mask=False):
    """
    Apply a mask to the input tensor where values are greater than a specified number of standard deviations from the mean.

    Args:
        tensor (torch.Tensor): The input tensor.
        factor (float): The number of standard deviations to use as the threshold.
        return_mask (bool, optional): If True, also return the mask used. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The masked tensor.
            - torch.Tensor: The mean of the mask values along the specified dimension.
            - torch.Tensor (optional): The mask used, if return_mask is True.
    """
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() > factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)

def less_than_std_mask(tensor, factor, return_mask=False):
    """
    Applies a mask to the input tensor where the mask is determined by whether 
    the absolute difference between each element and the mean of its row is less 
    than a specified factor times the standard deviation of its row.

    Args:
        tensor (torch.Tensor): The input tensor to be masked.
        factor (float): The factor by which the standard deviation is multiplied 
                        to determine the threshold for masking.
        return_mask (bool, optional): If True, the function also returns the mask 
                                      tensor. Default is False.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The masked tensor.
            - torch.Tensor: The mean of the mask values for each row.
            - torch.Tensor (optional): The mask tensor, if return_mask is True.
    """
    mask = (tensor - tensor.mean(dim=1).unsqueeze(1)).abs() < factor * tensor.std(
        dim=1
    ).unsqueeze(1)
    if return_mask:
        return tensor * mask, mask.float().mean(dim=1), mask
    return tensor * mask, mask.float().mean(dim=1)

def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    print(k)
    k = d - k  # Keep top k elements instead of bottom k elements
    print(k)
    k=max(min(k,d), 1)  # Ensure k is at least 1
    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values 
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask
    print(final_mask)
    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)

def bottomk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)

    # Create a mask tensor with True for the bottom k elements in each row
    mask = M.abs() <= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)

def topk_NFS_mask_preserve_normfrac(
    T, 
    normfrac=0.9, 
    self_cof=0.5, 
    similarity_cof=0.5, 
    return_mask=False
):
    """
    Pruning method.
    Args:
        T (torch.Tensor): Input tensor of shape (N, D) where N is the number of rows and D is the number of columns.
        normfrac (float): Fraction of the norm to preserve for each row (between 0 and 1).
        self_cof (float): Coefficient for self-contribution
        similarity_cof (float): Coefficient for alignment contribution
        return_mask (bool): If True, also return the mask used for pruning.
    Returns:
        torch.Tensor: Pruned tensor of the same shape as T.
        torch.Tensor: Keep ratio for each row.
        torch.Tensor (optional): Mask used for pruning, if return_mask is True.
    """
    # self contribution
    row_norms = torch.norm(T, p='fro', dim=1, keepdim=True)  # [N, 1]
    self_scores = (T.abs() ** 2) / (row_norms ** 2 + 1e-8)  # [N, D]

    # relevance contribution
    N, D = T.size()
    similarity_scores = torch.zeros_like(self_scores)
    for i in range(N):
        others = torch.cat([T[:i], T[i+1:]], dim=0)
        avg_other = others.mean(dim=0, keepdim=True)
        sim = (T[i:i+1] * avg_other).abs()
        sim_norm = torch.norm(avg_other, p='fro') * row_norms[i]
        similarity_scores[i] = sim / (sim_norm + 1e-8)
            
    self_scores_normalized = self_scores / (self_scores.sum(dim=1, keepdim=True) + 1e-8)
    similarity_scores_normalized = similarity_scores / (similarity_scores.sum(dim=1, keepdim=True) + 1e-8)

    # combination 
    scores = self_cof * self_scores + similarity_cof * similarity_scores  # [N, D]

  
    sorted_scores, sorted_indices = torch.sort(scores, dim=1, descending=True)
    sorted_self_scores = torch.gather(self_scores, 1, sorted_indices)
    cumsum_proportions = torch.cumsum(sorted_self_scores, dim=1)
    normfrac_mask = cumsum_proportions >= normfrac
    normfrac_indices = torch.argmax(normfrac_mask.float(), dim=1)  # [N]

    range_tensor = torch.arange(D, device=T.device).unsqueeze(0).expand(N, -1)
    mask = range_tensor <= normfrac_indices.unsqueeze(1)
    keep_ratio = mask.float().mean(dim=1)  
  
    final_mask = torch.zeros_like(T, dtype=torch.bool)
    final_mask.scatter_(1, sorted_indices, mask)

    if return_mask:
        return T * final_mask, keep_ratio, final_mask
    else:
        return T * final_mask, keep_ratio
 


def load_tv(merging_tasks,filepaths,ptm_path):
    # load the finetune and pretrained checkpoints
    ft_checks = [torch.load(fp,map_location="cpu") for fp in filepaths]
    # load the pretrained model
    ptm_check = torch.load(ptm_path,map_location="cpu")
    # check if all checkpoints have the same paramters.
    check_parameterNamesMatch(ft_checks + [ptm_check])
    remove_keys = []
    remove_keys= [ key for key in ptm_check.keys() if 'adapter' in key]
    reshape_keys = []
    for key in remove_keys + reshape_keys:
        if ptm_check[key].dim() == 1:
            ptm_check[key] = ptm_check[key][: len(ft_checks[0][key])]
        elif ptm_check[key].dim() == 2:
            ptm_check[key] = ptm_check[key][: len(ft_checks[0][key]), :]
        else:
            raise ValueError(f"Unexpected tensor dimension for key {key}: {ptm_check[key].dim()}")

    sd_to_tv_time = time.time()
    flat_ft = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    sd_to_tv_time = time.time() - sd_to_tv_time
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm

    
    # check if the vectorized state dicts can be converted back to the original state dicts.
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ft_checks[0], remove_keys), ptm_check
    )
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check
    )
    assert all(
        [
            check_state_dicts_equal(
                vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]
            )
            for i in range(len(ft_checks))
        ]
    )

    return (
        tv_flat_checks,
        flat_ptm,
        ptm_check,
        remove_keys,
        reshape_keys
    )
      



   
from typing import Optional

import numpy as np
import rsatoolbox as rsa

from .activations import get_activations_model, get_activations_allen, get_activations_pixel, get_activations_monkey
from .util import merge_dicts, load_object, save_object, get_save_path


def get_rdms_model(ckpt_paths: list, mode: str, seq_len: int, dest_folder=None, model_spec=None, **stim_kwargs):
    """
    Calculate RDMs for a list of model checkpoints.

    Parameters
    ----------
    ckpt_paths : list
        List of paths to the model checkpoints
    mode : str
        Which mode to use, can be one of 'allen', 'monkey'.
    seq_len : int
        Sequence length to use for the stimulus and model.
    dest_folder : str
        Folder to save the RDMs.
    model_spec : str
        Model specification with format '{species}_{nrb}', where species is the species of the model and nrb is the
        number of recurrent blocks in the model.
    stim_kwargs :
        Additional arguments for the stimulus dataset.

    Returns
    -------
    dict
        Dictionary of list of rdms for each layer in the models.
    """
    print(f"Calculating {mode} RDMs for {model_spec}")
    fname = f"{model_spec}" if model_spec is not None else "rdms"
    if mode == "allen":
        fname = f"{fname}_{stim_kwargs['stim_type']}"
    path = get_save_path(dest_folder, mode, 'rdms', fname, ext='pt')
    rdms = load_object(path)

    if rdms is None:
        print(f"Calculating RDMs for {mode} stimuli pixels")
        rdm_pixel = get_rdms_pixel(mode=mode, seq_len=seq_len, **stim_kwargs)
        rdms = [rdm_pixel] * len(ckpt_paths)
        for ckpt_path in ckpt_paths:
            act = get_activations_model(ckpt_path=ckpt_path, mode=mode, seq_len=seq_len, **stim_kwargs)
            print(f"Calculating RDMs")
            act_dset = {k: rsa.data.Dataset(v.flatten(1).numpy()) for k, v in act.items()}
            rdm = {k: rsa.rdm.calc_rdm(v, method="correlation") for k, v in act_dset.items()}
            rdms.append(rdm)

        rdms = merge_dicts(rdms)
        rdms = {k: rsa.rdm.concat(v) for k, v in rdms.items()}
        save_object(rdms, path)
    return rdms


def get_rdms_pixel(mode: str, seq_len: int, **stim_kwargs):
    """
    Calculate RDMs for the pixel representation of the stimulus.

    Parameters
    ----------
    mode : str
        Which mode to use, can be one of 'allen', 'monkey'.
    seq_len : int
        Sequence length to use for the stimulus.
    stim_kwargs :
        Additional arguments for the stimulus dataset.

    Returns
    -------
    dict
        Dictionary of RDM for the pixel representation of the stimulus.
    """
    act = get_activations_pixel(mode=mode, seq_len=seq_len, **stim_kwargs)
    act_dset = rsa.data.Dataset(act.flatten(1).numpy())
    rdm = rsa.rdm.calc_rdm(act_dset, method="correlation")
    return {'pixel': rdm}


def get_rdms_monkey(json_path_responses: str, area: str, num_iter: int, dest_folder=None):
    """
    Calculate RDMs for the monkey data.

    Parameters
    ----------
    json_path_responses : str
        Path to the monkey data.
    area : str
        Which brain area to use. Can be one of 'V1', 'V2'.
    num_iter : int
        Number of iterations to estimate the noise ceiling.
    dest_folder : str
        Folder to save the RDMs.

    Returns
    -------
    dict
        Dictionary of RDMs for the monkey data.
    np.ndarray
        Noise ceiling for the RDMs.
    """
    print(f"Calculating monkey RDMs for {area}")
    path_nc = get_save_path(dest_folder, 'monkey', 'noise_ceiling', f'{area}', ext='pt')
    noise_ceiling = load_object(path_nc)

    path = get_save_path(dest_folder, 'monkey', 'rdms', f'{area}', ext='pt')
    rdms = load_object(path)

    if noise_ceiling is None or rdms is None:
        activations = get_activations_monkey(json_path=json_path_responses, area=area)
        rdms, noise_ceiling = get_rdms_brain(activations, num_iter)
        save_object(noise_ceiling, path_nc)
        save_object(rdms, path)
    return rdms, noise_ceiling


def get_rdms_allen(area: str, depth: int, cre_line: str, stim_type: str, num_iter: int, seq_len: int, dest_folder=None):
    """
    Calculate RDMs for the Allen Brain Observatory data.

    Parameters
    ----------
    area : str
        Which brain area to use. Can be one of 'VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl'.
    depth : int
        Which depth to use.
    cre_line : str
        Which cre line to use.
    stim_type : str
        Which stimulus type to use, can be one of 'natural_scenes', 'natural_movie_one', 'natural_movie_two',
        'natural_movie_three'.
    num_iter : int
        Number of iterations to estimate the noise ceiling.
    seq_len : int
        Sequence length to use for the stimulus.
    dest_folder : str
        Folder to save the RDMs.

    Returns
    -------
    dict
        Dictionary of RDMs for the Allen Brain Observatory data.
    np.ndarray
        Noise ceiling for the RDMs.
    """
    print(f"Calculating allen RDMs for {stim_type} {area} {depth} {cre_line}")
    path_nc = get_save_path(dest_folder, 'allen', 'noise_ceiling', f'{stim_type}_{area}_{depth}_{cre_line}', ext='pt')
    noise_ceiling = load_object(path_nc)

    path = get_save_path(dest_folder, 'allen', 'rdms', f'{stim_type}_{area}_{depth}_{cre_line}', ext='pt')
    rdms = load_object(path)

    if noise_ceiling is None or rdms is None:
        activations = get_activations_allen(area=area, depth=depth, cre_line=cre_line, stim_type=stim_type,
                                            seq_len=seq_len)
        rdms, noise_ceiling = get_rdms_brain(activations, num_iter)
        save_object(noise_ceiling, path_nc)
        save_object(rdms, path)
    return rdms, noise_ceiling


def get_rdms_brain(activations, num_iter):
    """
    Calculate RDMs for the brain data.

    Parameters
    ----------
    activations : np.ndarray
        Activations of shape T x M x N, where T is the number of trials, M is the number of stimuli, and N is the number
        of neurons.
    num_iter : int
        Number of iterations to estimate the noise ceiling.

    Returns
    -------
    dict
        Dictionary of RDMs for the Allen Brain Observatory data.
    np.ndarray
        Noise ceiling for the RDMs.
    """
    num_trials = activations.shape[0]
    print("Calculating noise ceiling")
    noise_ceiling = estimate_noise_ceiling(activations=activations,
                                           num_iter=num_iter if num_iter is not None else num_trials)
    print("Calculating RDMs")
    rdms = bootstrap_rdms(activations, num_iter=num_iter)
    return rdms, noise_ceiling


def bootstrap_rdms(activations: np.ndarray, num_iter: Optional[int] = None):
    """
    Bootstrap RDMs.

    Parameters
    ----------
    activations : np.ndarray
        Activations to bootstrap.
    num_iter : int
        Number of iterations to bootstrap.

    Returns
    -------
    np.ndarray
        Bootstrapped RDMs.
    """
    num_trials = activations.shape[0]
    rdms = []
    if num_iter is not None:
        for i in range(num_iter):
            trials_idx_permute = np.random.permutation(np.arange(num_trials))
            which_trials = trials_idx_permute[:num_trials // 2]

            act = activations[which_trials].mean(0)
            act_dset = rsa.data.Dataset(act)
            rdm = rsa.rdm.calc_rdm(act_dset, method="correlation")
            rdms.append(rdm)
    else:
        for act in activations:
            act_dset = rsa.data.Dataset(act)
            rdm = rsa.rdm.calc_rdm(act_dset, method="correlation")
            rdms.append(rdm)

    return rsa.rdm.concat(rdms)


def estimate_noise_ceiling(activations: np.ndarray, num_iter: int):
    """
    Estimate the noise ceiling for the RDMs.

    Parameters
    ----------
    activations : np.ndarray
        Activations of shape T x M x N, where T is the number of trials, M is the number of stimuli, and N is the number
        of neurons.
    num_iter : int
        Number of iterations to estimate the noise ceiling.

    Returns
    -------
    np.ndarray
        Noise ceiling for the RDMs.
    """
    num_trials = activations.shape[0]
    r1 = []
    for i in range(num_iter):
        trials_idx_permute = np.random.permutation(np.arange(num_trials))
        which_trials = trials_idx_permute[:num_trials // 2]
        other_trials = trials_idx_permute[num_trials // 2:]

        responses1 = activations[which_trials, :, :].mean(0)
        responses2 = activations[other_trials, :, :].mean(0)

        responses1 = rsa.data.Dataset(responses1)
        responses2 = rsa.data.Dataset(responses2)

        rdm1 = rsa.rdm.calc_rdm(responses1, method="correlation").get_matrices()[0]
        rdm2 = rsa.rdm.calc_rdm(responses2, method="correlation").get_matrices()[0]

        np.fill_diagonal(rdm1, 'nan')
        np.fill_diagonal(rdm2, 'nan')

        rdm1 = rsa.rdm.RDMs(rdm1)
        rdm2 = rsa.rdm.RDMs(rdm2)

        r1.append(rsa.rdm.compare(rdm1, rdm2, method="tau-a")[0][0])

    return np.array(r1)

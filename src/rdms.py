import numpy as np
import rsatoolbox as rsa

from .activations import get_activations_model, get_activations_allen, get_activations_pixel
from .util import merge_dicts, cat_rdms


def get_rdms_model(ckpt_paths: list, stim_type: str, seq_len: int):
    """
    Calculate RDMs for a list of model checkpoints.

    Parameters
    ----------
    ckpt_paths : list
        List of paths to the model checkpoints
    stim_type : str
        Which stimulus type to use, can be one of 'natural_scenes', 'natural_movie_one', 'natural_movie_two',
        'natural_movie_three'.
    seq_len : int
        Sequence length to use for the stimulus and model.

    Returns
    -------
    dict
        Dictionary of list of rdms for each layer in the models.
    """
    print(f"Calculating RDMs for pixels")
    rdm_pixel = get_rdms_pixel(stim_type=stim_type, seq_len=seq_len)
    rdms = [rdm_pixel] * len(ckpt_paths)

    for ckpt_path in ckpt_paths:
        print(f"Calculating RDMs for {ckpt_path}")
        act = get_activations_model(ckpt_path=ckpt_path, stim_type=stim_type, seq_len=seq_len)
        act_dset = {k: rsa.data.Dataset(v.flatten(1).numpy()) for k, v in act.items()}
        rdm = {k: rsa.rdm.calc_rdm(v, method="correlation") for k, v in act_dset.items()}
        rdms.append(rdm)

    rdms = merge_dicts(rdms)
    rdms = {k: cat_rdms(v) for k, v in rdms.items()}
    return rdms


def get_rdms_pixel(stim_type: str, seq_len: int):
    """
    Calculate RDMs for the pixel representation of the stimulus.

    Parameters
    ----------
    stim_type : str
        Which stimulus type to use, can be one of 'natural_scenes', 'natural_movie_one', 'natural_movie_two',
        'natural_movie_three'.
    seq_len : int
        Sequence length to use for the stimulus.

    Returns
    -------
    dict
        Dictionary of RDM for the pixel representation of the stimulus.
    """
    act = get_activations_pixel(stim_type=stim_type, seq_len=seq_len)
    act_dset = rsa.data.Dataset(act.flatten(1).numpy())
    rdm = rsa.rdm.calc_rdm(act_dset, method="correlation")
    return {'pixel': rdm}


def get_rdms_allen(area: str, depth: int, cre_line: str, stim_type: str, num_iter: int, seq_len: int):
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

    Returns
    -------
    dict
        Dictionary of RDMs for the Allen Brain Observatory data.
    np.ndarray
        Noise ceiling for the RDMs.
    """
    print(f"Calculating RDMs for {stim_type} {area} {depth} {cre_line}")
    activations = get_activations_allen(area=area, depth=depth, cre_line=cre_line, stim_type=stim_type,
                                        seq_len=seq_len)
    num_trials = activations.shape[0]
    noise_ceiling = estimate_rdm_noise_ceiling(activations=activations,
                                               num_iter=num_iter if num_iter is not None else num_trials)

    rdms = []
    if num_iter is not None:
        for i in range(num_iter):
            trials_idx_permute = np.random.permutation(np.arange(0, num_trials))
            which_trials = trials_idx_permute[0:num_trials // 2]

            act = activations[which_trials].mean(0)
            act_dset = rsa.data.Dataset(act)
            rdm = rsa.rdm.calc_rdm(act_dset, method="correlation")
            rdms.append(rdm)
    else:
        for act in activations:
            act_dset = rsa.data.Dataset(act)
            rdm = rsa.rdm.calc_rdm(act_dset, method="correlation")
            rdms.append(rdm)

    rdms = cat_rdms(rdms)
    return rdms, noise_ceiling


def estimate_rdm_noise_ceiling(activations: np.ndarray, num_iter: int):
    """
    Estimate the noise ceiling for the RDMs.

    Parameters
    ----------
    activations : np.ndarray
        Activations of shape T x M x N, where T is the number of trials, M is the number of stimuli, and N is the number
        of neurons.
    num_iter : int
        Number of iterations to estimate the noise ceiling.
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

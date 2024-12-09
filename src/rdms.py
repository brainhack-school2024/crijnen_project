from typing import Optional

import numpy as np
import rsatoolbox as rsa

from .util import load_object, save_object, get_save_path, set_seed


class BaseRDMs:
    def __init__(self, dest_folder: Optional[str] = None, seed: Optional[int] = None, **kwargs):
        self.activations = {}
        self.rdms = {}

        self.seed = seed
        set_seed(seed)
        self.dest_folder = get_save_path(dest_folder, f'seed_{seed}', ext=None)

    def get_activations(self, *args, **kwargs):
        raise NotImplementedError

    def calculate_rdms(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, root: str, fname: str, key, *objects):
        for obj in objects:
            path = get_save_path(root, obj, fname, ext='pt')
            loaded = load_object(path)
            if loaded is not None:
                print(f'Loaded {obj}')
            self.__getattribute__(obj)[key] = loaded

    @staticmethod
    def save(dest_folder: str, fname: str, **objects):
        for k, v in objects.items():
            path = get_save_path(dest_folder, k, fname, ext='pt')
            save_object(v, path)


class BrainRDMs(BaseRDMs):
    def __init__(self, dest_folder: Optional[str] = None, seed: Optional[int] = None, **kwargs):
        super().__init__(dest_folder=dest_folder, seed=seed)
        self.noise_ceilings = {}

    @staticmethod
    def calculate_rdms(
            activations: np.ndarray,
            num_iter: Optional[int] = None,
            seed: Optional[int] = None
    ):
        return bootstrap_rdms(activations, num_iter=num_iter, seed=seed)

    @staticmethod
    def estimate_noise_ceiling(
            activations: np.ndarray,
            num_iter: Optional[int] = None,
            seed: Optional[int] = None
    ):
        return estimate_noise_ceiling(activations, num_iter=num_iter, seed=seed)


def bootstrap_rdms(activations: np.ndarray, num_iter: Optional[int] = None, seed: Optional[int] = None):
    """
    Bootstrap RDMs.

    Parameters
    ----------
    activations : np.ndarray
        Activations to bootstrap.
    num_iter : int
        Number of iterations to bootstrap.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Bootstrapped RDMs.
    """
    set_seed(seed)

    num_trials = activations.shape[0]
    if num_iter is not None:
        rdms = []
        for i in range(num_iter):
            trials_idx_permute = np.random.permutation(np.arange(num_trials))
            which_trials = trials_idx_permute[:num_trials // 2]

            act = activations[which_trials].mean(0)
            act_dset = rsa.data.Dataset(act)
            rdm = rsa.rdm.calc_rdm(act_dset, method="correlation")
            rdms.append(rdm)
        rdms = rsa.rdm.concat(rdms)
    else:
        act_dset = rsa.data.Dataset(activations.mean(0))
        rdms = rsa.rdm.calc_rdm(act_dset, method="correlation")

    return rdms


def estimate_noise_ceiling(activations: np.ndarray, num_iter: Optional[int] = None, seed: Optional[int] = None):
    """
    Estimate the noise ceiling for the RDMs.

    Parameters
    ----------
    activations : np.ndarray
        Activations of shape T x M x N, where T is the number of trials, M is the number of stimuli, and N is the number
        of neurons.
    num_iter : int
        Number of iterations to estimate the noise ceiling.
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray
        Noise ceiling for the RDMs.
    """
    set_seed(seed)

    num_trials = activations.shape[0]
    num_iter = num_iter if num_iter is not None else num_trials
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

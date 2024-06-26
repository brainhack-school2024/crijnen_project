import rsatoolbox as rsa
import os
import torch
import numpy as np

from src.rdms import get_rdms_allen, get_rdms_model
from src.util import plot_rsa


def compare_rsa(models: dict, areas: list = [('VISp', 275, 'Cux2-CreERT2')], stim_type: str = 'natural_movie_one',
                num_iter: int = 100, seq_len: int = 15, fig_path: str = None, seed: int = 42):
    """
    Compare RSA between Allen Brain Observatory data and model activations.

    Parameters
    ----------
    models : dict
        Dictionary of species and their model checkpoint paths.
    areas : list
        List of tuples of brain areas, depth, and cre line to use.
    stim_type : str
        Which stimulus type to use, can be one of 'natural_scenes', 'natural_movie_one', 'natural_movie_two',
        'natural_movie_three'.
    num_iter : int
        Number of iterations to use for noise ceiling estimation using bootstrap.
    seq_len : int
        Sequence length to use for the stimulus and model.
    fig_path : str
        Path to save the figures.
    seed : int
        Random seed to use for reproducibility.
    """
    np.random.seed(seed)
    all_rsas = {}
    all_rdms = {}
    all_nc = {}
    for a in areas:
        area, depth, cre_line = a

        path = os.path.join(fig_path if fig_path is not None else '', area, stim_type, f'{depth}_{cre_line}_{seed}.pt')
        if os.path.exists(path):
            data = torch.load(path)
            rsas, rdms, noise_ceiling = data['rsa'], data['rdm'], data['nc']
        else:
            rsas, rdms, noise_ceiling = calculate_rsa(area=area, depth=depth, cre_line=cre_line, stim_type=stim_type,
                                                      num_iter=num_iter, models=models, seq_len=seq_len)

        save_path = fig_path
        if save_path is not None:
            out_dir = os.path.join(save_path, area, stim_type)
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f'{depth}_{cre_line}')
            if not os.path.exists(path):
                torch.save({'rsa': rsas, 'rdm': rdms, 'nc': noise_ceiling}, path)

        plot_rsa(rsas, area=a, noise_ceiling=noise_ceiling, stim_type=stim_type, noise_corrected=False,
                 save_path=save_path)
        plot_rsa(rsas, area=a, noise_ceiling=noise_ceiling, stim_type=stim_type, noise_corrected=True,
                 save_path=save_path)

        all_rsas[a] = rsas
        all_rdms[a] = rdms
        all_nc[a] = noise_ceiling

    return all_rsas, all_rdms, all_nc


def calculate_rsa(area: str, depth: int, cre_line: str, stim_type: str, num_iter: int, models: dict, seq_len: int):
    """
    Calculate rdm similarity between Allen Brain Observatory data and model activations using kendall's tau-a.

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
        Number of iterations to use for noise ceiling estimation using bootstrap.
    models : dict
        Dictionary of species and their model checkpoint paths.
    seq_len : int
        Sequence length to use for the stimulus.

    Returns
    -------
    dict
        Dictionary of RSA values for each model.
    dict
        Dictionary of RDMs for each model.
    np.ndarray
        Noise ceiling for the Allen Brain Observatory data.
    """
    rdms_allen, noise_ceiling = get_rdms_allen(area=area, depth=depth, cre_line=cre_line, stim_type=stim_type,
                                               num_iter=num_iter, seq_len=seq_len)
    all_rsas = {}
    all_rdms = {"allen": rdms_allen}
    for species, ckpt_paths in models.items():
        print(f"{species} RSA")
        rdms = get_rdms_model(ckpt_paths, stim_type=stim_type, seq_len=seq_len)
        print(f"Calculating RDM similarity between {area} and {species}")
        rsas = {k: rsa.rdm.compare(rdms_allen, rdm, method='tau-a').mean(0) for k, rdm in rdms.items()}
        all_rdms[species] = rdms
        all_rsas[species] = rsas
        print("Done\n")

    return all_rsas, all_rdms, noise_ceiling

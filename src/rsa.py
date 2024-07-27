import numpy as np
import rsatoolbox as rsa
import os

from src.rdms import get_rdms_allen, get_rdms_monkey, get_rdms_model
from src.util import get_save_path, load_object, save_object, plot_rsa


def compare_rsa(models: dict, config: dict, num_iter: int = 100, seq_len: int = 15, dest_folder: str = None,
                seed: int = 42):
    """
    Compare RSA between Allen Brain Observatory data and model activations.

    Parameters
    ----------
    models : dict
        Dictionary of species and their model checkpoint paths.
    config : dict
        List of tuples of brain areas, depth, and cre line to use.
    num_iter : int
        Number of iterations to use for noise ceiling estimation using bootstrap.
    seq_len : int
        Sequence length to use for the stimulus and model.
    dest_folder : str
        Folder to save the results.
    seed : int
        Random seed to use for reproducibility.
    """
    np.random.seed(seed)
    dest_folder = get_save_path(dest_folder, f'seed_{seed}', ext=None)
    print(f"Saving results to {dest_folder}")
    if config is None:
        config = {
            'allen': {
                'stim_types': ['natural_movie_one'],
                'areas': [('VISp', 275, 'Cux2-CreERT2')],
            },
            'monkey': {
                'json_path_responses': '../data/movshon_benchmark/responses/raw/not_averaged/MovshonFreemanZiemba2013_responses.json',
                'json_path_images': '../data/movshon_benchmark/responses/raw/not_averaged/MovshonFreemanZiemba2013_orders.json',
                'img_root': '../data/movshon_benchmark/images/image_movshon_FreemanZiemba2013-public',
                'areas': ['V1', 'V2']
            }
        }

    all_rsas = {k: {} for k in config.keys()}
    all_rdms = {k: {} for k in config.keys()}
    all_nc = {k: {} for k in config.keys()}

    for mode, mode_config in config.items():
        areas = mode_config['areas']
        if mode == 'allen':
            stim_types = mode_config['stim_types']
            for stim_type in stim_types:
                for a in areas:
                    area, depth, cre_line = a
                    print(dest_folder)
                    rsas, rdms, noise_ceiling = calculate_rsa_allen(area=area, depth=depth, cre_line=cre_line,
                                                                    stim_type=stim_type, num_iter=num_iter,
                                                                    models=models, seq_len=seq_len,
                                                                    dest_folder=dest_folder)

                    plot_config = {
                        'mode': mode,
                        'area': area,
                        'depth': depth,
                        'cre_line': cre_line,
                        'stim_type': stim_type
                    }
                    save_path = get_save_path(dest_folder, mode, 'plots', f'{stim_type}_{area}_{depth}_{cre_line}',
                                              ext=None)
                    plot_rsa(rsas, config=plot_config.copy(), noise_ceiling=noise_ceiling, noise_corrected=False,
                             save_path=save_path)
                    plot_rsa(rsas, config=plot_config.copy(), noise_ceiling=noise_ceiling, noise_corrected=True,
                             save_path=save_path)

                    all_rsas[mode][f'{stim_type}_{area}_{depth}_{cre_line}'] = rsas
                    all_rdms[mode][f'{stim_type}_{area}_{depth}_{cre_line}'] = rdms
                    all_nc[mode][f'{stim_type}_{area}_{depth}_{cre_line}'] = noise_ceiling

        elif mode == 'monkey':
            json_path_responses = mode_config['json_path_responses']
            json_path_images = mode_config['json_path_images']
            img_root = mode_config['img_root']

            for area in areas:
                rsas, rdms, noise_ceiling = calculate_rsa_monkey(json_path_responses=json_path_responses,
                                                                 json_path_images=json_path_images, img_root=img_root,
                                                                 area=area, num_iter=num_iter, models=models,
                                                                 seq_len=seq_len, dest_folder=dest_folder)

                plot_config = {
                    'mode': mode,
                    'area': area
                }
                save_path = get_save_path(dest_folder, mode, 'plots', area, ext=None)
                plot_rsa(rsas, config=plot_config.copy(), noise_ceiling=noise_ceiling, noise_corrected=False,
                         save_path=save_path)
                plot_rsa(rsas, config=plot_config.copy(), noise_ceiling=noise_ceiling, noise_corrected=True,
                         save_path=save_path)

                all_rsas[mode][f'{area}'] = rsas
                all_rdms[mode][f'{area}'] = rdms
                all_nc[mode][f'{area}'] = noise_ceiling

    return all_rsas, all_rdms, all_nc


def calculate_rsa_allen(area: str, depth: int, cre_line: str, stim_type: str, num_iter: int,
                        models: dict, seq_len: int, dest_folder: str):
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
    dest_folder : str
        Folder to save the RSA and RDMs.

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
                                               num_iter=num_iter, seq_len=seq_len, dest_folder=dest_folder)
    all_rsas = {}
    all_rdms = {"allen": rdms_allen}
    for model_spec, ckpt_paths in models.items():
        print(f"{model_spec} RSA")
        rdms = get_rdms_model(ckpt_paths=ckpt_paths, mode='allen', seq_len=seq_len,
                              dest_folder=dest_folder, model_spec=model_spec,
                              stim_type=stim_type)

        print(f"Calculating RDM similarity between mouse {area} and {model_spec}")
        path = get_save_path(dest_folder, 'allen', 'rsa', f'{model_spec}_{area}_{depth}_{cre_line}_{stim_type}',
                             ext='pt')
        rsas = load_object(path)

        if rsas is None:
            rsas = {k: rsa.rdm.compare(rdms_allen, rdm, method='tau-a').mean(0) for k, rdm in rdms.items()}
            save_object(rsas, path)

        all_rdms[model_spec] = rdms
        all_rsas[model_spec] = rsas
        print("Done\n")

    return all_rsas, all_rdms, noise_ceiling


def calculate_rsa_monkey(json_path_responses: str, json_path_images: str, img_root: str, area: str, num_iter: int,
                         models: dict, seq_len: int, dest_folder: str):
    """
    Calculate rdm similarity between Allen Brain Observatory data and model activations using kendall's tau-a.

    Parameters
    ----------
    json_path_responses : str
        Path to the json file containing the monkey responses.
    json_path_images : str
        Path to the json file containing the image order.
    img_root : str
        Root folder containing the images.
    area : str
        Which brain area to use. Can be one of 'V1', 'V2'.
    num_iter : int
        Number of iterations to use for noise ceiling estimation using bootstrap.
    models : dict
        Dictionary of species and their model checkpoint paths.
    seq_len : int
        Sequence length to use for the stimulus.
    dest_folder : str
        Folder to save the RSA and RDMs.

    Returns
    -------
    dict
        Dictionary of RSA values for each model.
    dict
        Dictionary of RDMs for each model.
    np.ndarray
        Noise ceiling for the Allen Brain Observatory data.
    """
    rdms_monkey, noise_ceiling = get_rdms_monkey(json_path_responses=json_path_responses, area=area, num_iter=num_iter,
                                                 dest_folder=dest_folder)
    all_rsas = {}
    all_rdms = {"monkey": rdms_monkey}
    for model_spec, ckpt_paths in models.items():
        print(f"{model_spec} RSA")
        rdms = get_rdms_model(ckpt_paths=ckpt_paths, mode='monkey', seq_len=seq_len,
                              dest_folder=dest_folder, model_spec=model_spec,
                              json_path_images=json_path_images, root=img_root)

        print(f"Calculating RDM similarity between monkey {area} and {model_spec}")
        path = get_save_path(dest_folder, 'monkey', 'rsa', f'{model_spec}_{area}', ext='pt')
        rsas = load_object(path)

        if rsas is None:
            rsas = {k: rsa.rdm.compare(rdms_monkey, rdm, method='tau-a').mean(0) for k, rdm in rdms.items()}
            save_object(rsas, path)

        all_rdms[model_spec] = rdms
        all_rsas[model_spec] = rsas
        print("Done\n")

    return all_rsas, all_rdms, noise_ceiling

import os
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def plot_rsa(rsa: dict, config: dict, noise_ceiling: Optional[np.ndarray] = None, save_path: Optional[str] = None):
    """
    Plot RSA results.
    """
    mode = config.pop('mode')
    area = config.pop('area')
    stim_type = config.get('stim_type', None)
    depth = config.get('depth', None)
    cre_line = config.get('cre_line', None)

    data = []
    for species, kt in rsa.items():
        model, nrb_ = species.split('_')
        df_model = pd.DataFrame(kt).melt(var_name='Layer', value_name='RSA')
        if noise_ceiling is not None:
            df_model['RSA'] /= np.median(noise_ceiling)
        df_model['Model'] = model.capitalize()
        df_model['Nrb'] = int(nrb_)
        df_model['Path'] = [layer[:2] if layer[:2] in ['p1', 'p2'] else '' for layer in df_model['Layer']]
        df_model['Layer'] = [layer[3:] if layer[:2] in ['p1', 'p2'] else layer for layer in df_model['Layer']]
        data.append(df_model)

    df = pd.concat(data)
    df = df.drop_duplicates(keep='first')
    baseline = df.loc[df['Layer'] == 'pixel']['RSA'].drop_duplicates(keep='first').reset_index(drop=True)
    df = df.loc[df.Path != 'p2']
    df = df.loc[df['Layer'] != 'pixel']
    df.loc[df['Model'] == 'James', 'Model'] = 'Treeshrew2'

    for nrb in df.Nrb.unique():
        df2 = df.loc[df.Nrb == nrb]

        custom_order = ['s1'] + [f'res_block{i}' for i in range(1, nrb + 1)] + ['out']
        df2['Layer'] = pd.Categorical(df2['Layer'], categories=custom_order, ordered=True)
        df2 = df2.sort_values(['Model', 'Layer']).reset_index(drop=True)

        plt.figure(figsize=(12, 6))
        colors = ['black', 'red', 'darkblue', 'blue']
        sns.pointplot(data=df2, x='Layer', y='RSA', hue='Model', palette=colors, dodge=0.2, estimator='median',
                      err_kws={'linewidth': 1}, markersize=2, linestyle='none')
        x_limits = plt.xlim()
        plt.axhline(baseline[0], color='black', linestyle='--', label='Pixel')
        plt.xlabel('Layers')
        plt.xticks(rotation=45)
        plt.ylabel('RDM similarity\n(noise corrected)' if noise_ceiling is not None else
                   'RDM similarity')
        plt.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.xlim(x_limits)

        if save_path is not None:
            path = save_path + f'_{nrb}.svg'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path)

        title = f'{mode.capitalize()} {area}, NRB: {nrb}'
        title += f'\nStimulus: {stim_type}, Depth: {depth}, Cre line: {cre_line}' if mode == 'allen' else ''
        plt.title(title)
        plt.show()
    return df


def set_seed(seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_save_path(dest_folder, *path, ext='pt'):
    if dest_folder is not None:
        path = os.path.join(dest_folder, *path)
        return path if ext is None else f'{path}.{ext}'
    return None


def load_object(path: str):
    """
    Load RDMs from a file.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    object
        Loaded object.
    """
    if path is None:
        return None
    if os.path.exists(path):
        return torch.load(path)
    else:
        return None


def save_object(data: object, path: str):
    """
    Save object to a file.

    Parameters
    ----------
    data : object
        Object to save.
    path : str
        Path to save the object.
    """
    if path is not None and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data, path)


def merge_dicts(dicts: list):
    """
    Merge dictionaries creating a list of values for each key.

    Parameters
    ----------
    dicts : list
        List of dictionaries to merge.

    Returns
    -------
    dict
        Merged dictionary.
    """
    super_dict = {}
    for d in dicts:
        for k, v in d.items():
            super_dict.setdefault(k, []).append(v)
    return super_dict

import os
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorchvideo.transforms as TV
import seaborn as sns
import torch
import torchvision.transforms as T
from PIL import Image
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from dpc.models.dpc_plus_lit import DPCPlusLit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models._utils import IntermediateLayerGetter

boc = BrainObservatoryCache(
    manifest_file='../data/brain_observatory_manifest.json')


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
    dict
        Loaded RDMs.
    """
    if os.path.exists(path):
        return torch.load(path)
    else:
        return None


def save_object(data: object, path: str):
    """
    Save RDMs to a file.

    Parameters
    ----------
    data : dict
        RDMs to save.
    path : str
        Path to save the RDMs.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path is not None and not os.path.exists(path):
        torch.save(data, path)


def plot_rsa(rsa: dict, config: dict, noise_ceiling: Optional[np.ndarray] = None,
             noise_corrected: Optional[bool] = False, save_path: Optional[str] = None):
    """
    Plot RSA results.
    """
    mode = config.pop('mode')
    area = config.pop('area')
    data = []
    # max_layers = max([int(k.split('_')[-1]) for k in rsa.keys()])
    for species, kt in rsa.items():
        df_model = pd.DataFrame(kt).melt(var_name='Layer', value_name='RSA')
        df_model['Model'] = species
        df_model['Path'] = [layer[:2] if layer[:2] in ['p1', 'p2'] else '' for layer in df_model['Layer']]
        df_model['Layer'] = [layer[3:] if layer[:2] in ['p1', 'p2'] else layer for layer in df_model['Layer']]
        data.append(df_model)

    df = pd.concat(data)
    df['Model_Path'] = df['Model'] + df['Path'].replace({'': '', 'p1': ' P1', 'p2': ' P2'})
    if noise_ceiling is not None and noise_corrected:
        df['RSA'] /= np.median(noise_ceiling)

    plt.figure(figsize=(18, 6))
    colors = ['darkblue'] * 3 + ['darkred'] * 3 + ['blue'] * 3 + ['red'] * 3 + ['deepskyblue'] * 3 + ['orange'] * 3
    pointplot = sns.pointplot(data=df, x='Layer', y='RSA', hue='Model_Path', palette=colors, dodge=0.7,
                              estimator='median', errwidth=1, markersize=2, markers=['o', '<', '>'] * 6, join=False)
    x_limits = plt.xlim()
    if noise_ceiling is not None and not noise_corrected:
        med = np.median(noise_ceiling)
        sd = np.std(noise_ceiling)
        plt.axhline(med, color='black', linestyle='--', label='Noise Ceiling Median')
        plt.fill_between(x=[x_limits[0], x_limits[1]], y1=med-sd, y2=med+sd, color='gray', alpha=0.5,
                         label='Noise Ceiling SD')
    plt.xlabel('Layers')
    plt.xticks(rotation=45)
    plt.ylabel('RDM similarity\n(noise corrected)' if noise_ceiling is not None and noise_corrected else
               'RDM similarity')
    plt.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(x_limits)

    if save_path is not None:
        save_path = save_path + ('_nc' if noise_ceiling is not None and noise_corrected else '') + '.svg'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    plt.title(f'Representational Similarity Analysis between {mode} {area} and ANNs trained with SSL'
              '\n' if len(config) > 0 else ''
              f', '.join([f'{k}: {v}' for k, v in config.items()]) if len(config) > 0 else '')
    plt.show()
    return df


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


def load_model(ckpt_path: str):
    """
    Load model from checkpoint.

    Parameters
    ----------
    ckpt_path : str
        Path to the model checkpoint.

    Returns
    -------
    nn.Module
        Loaded model.
    dict
        Normalization values (mean and std).
    int
        Sequence length of the model.
    """
    model = DPCPlusLit.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    model.freeze()
    norm_kwargs = {"mean": model.hparams.mean, "std": model.hparams.std}
    seq_len = model.network.seq_len
    model = Model(model)
    return model, norm_kwargs, seq_len


def extract_one_path(model: nn.Module, path: int):
    """
    Extract one path from the model using IntermediateLayerGetter which returns the intermediate activations.

    Parameters
    ----------
    model : nn.Module
        Model to extract path from.
    path : int
        Path to extract.

    Returns
    -------
    IntermediateLayerGetter
        Extracted path
    """
    m = nn.Sequential()
    for i, block in enumerate(model.network.backbone.get_submodule(f"path{path}").res_blocks.children()):
        m.add_module(f"p{path + 1}_res_block{i + 1}", block)
    m.add_module("out", model.network.backbone.get_submodule(f"dropout{path}"))

    layers = [n for n, _ in m.named_children()]
    m = IntermediateLayerGetter(m, dict(zip(layers, layers)))
    return m


def get_stimulus(mode: str, seq_len: int, norm_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
    """
    Get stimulus dataset.

    Parameters
    ----------
    mode : str
        Which stimulus dataset to use, can be one of 'allen', 'monkey'.
    seq_len : int
        Sequence length to use for the stimulus.
    norm_kwargs : dict
        Normalization values (mean and std).
    **kwargs
        Additional arguments for the stimulus dataset.

    Returns
    -------
    DataLoader
        Stimulus DataLoader.
    """
    if mode == "allen":
        data = get_stimulus_allen(seq_len=seq_len, **kwargs)
    elif mode == "monkey":
        data = get_stimulus_monkey(**kwargs)
    else:
        raise ValueError(f'Invalid mode {mode}')

    transforms = T.Compose([
        T.ToTensor(),
        T.Resize((64, 64), antialias=True),
        T.Lambda(lambda x: x.expand(3, seq_len, -1, -1)),
        TV.Normalize(**norm_kwargs) if norm_kwargs is not None else T.Lambda(lambda x: x),
    ])

    ds = StimuliDataset(data, transform=transforms)
    return DataLoader(ds, batch_size=32, num_workers=8, shuffle=False)


def get_stimulus_allen(stim_type: str, seq_len: int):
    """
    Get stimulus data.

    Parameters
    ----------
    stim_type : str
        Which stimulus type to use, can be one of 'natural_scenes', 'natural_movie_one', 'natural_movie_two',
        'natural_movie_three'.
    seq_len : int
        Sequence length to use for the stimulus.

    Returns
    -------
    DataLoader
        Stimulus DataLoader.
    """
    data_set = boc.get_ophys_experiment_data(501498760)
    data = data_set.get_stimulus_template(stim_type)

    _, h, w = data.shape
    if stim_type == "natural_scenes":
        return data[:, :, :, None]
    elif stim_type in ['natural_movie_one', 'natural_movie_two', 'natural_movie_three']:
        return data.reshape(-1, seq_len, h, w).transpose(0, 2, 3, 1)
    else:
        raise ValueError(f'Invalid stimulus type {stim_type}')


def get_stimulus_monkey(json_path_images: str, root: str):
    images = pd.read_json(json_path_images)
    images = images.drop_duplicates(subset='Image_file', keep='first')

    data = []
    for img_path in images.Image_file:
        img_path = os.path.join(root, img_path)
        with open(img_path, "rb") as f:
            img = Image.open(f)
            data.append(np.array(img, np.uint8, copy=True))

    return np.array(data)[:, :, :, None]


class StimuliDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.images = dataset
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.s1 = model.network.backbone.s1
        self.num_paths = model.network.num_paths
        for i in range(self.num_paths):
            path = extract_one_path(model, i)
            self.add_module(f"path{i}", path)
        self.layers = [
            "s1",
            *[n for i in range(self.num_paths) for n, _ in self.get_submodule(f"path{i}").named_children() if
              n != "out"],
            "out"
        ]

    def forward(self, x):
        s1 = self.s1(x)
        paths = [self.get_submodule(f"path{i}")(s1) for i in range(self.num_paths)]
        outs = [p.pop("out") for p in paths]

        r = {"s1": s1}
        for d in paths:
            r.update(d)

        if isinstance(self.model.network.concat, bool) and self.model.network.concat:
            r["out"] = torch.cat(outs, dim=1)
        else:
            r["out"] = sum(outs)

        return r

import os
from typing import Optional, Any, List, Dict

import numpy as np
import pandas as pd
import pytorchvideo.transforms as TV
import rsatoolbox as rsa
import torch
import torchvision.transforms as T
from PIL import Image
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from dpc.models.dpc_plus_lit import DPCPlusLit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models._utils import IntermediateLayerGetter

from .rdms import BaseRDMs
from .rdms_monkey import image_order_majaj
from .util import merge_dicts, get_save_path


class ModelRDMs(BaseRDMs):
    def __init__(
            self,
            stimuli: Dict[str, dict],
            ckpt_paths: List[str],
            name: str,
            dest_folder: Optional[str] = None,
            force_activations: Optional[bool] = False,
            seed: Optional[int] = None,
            **kwargs
    ):
        super().__init__(dest_folder=dest_folder, seed=seed)
        print(f'Processing model {name}')

        models, norm_kwargs, seq_len = load_models(ckpt_paths)

        for dset, config in stimuli.items():
            dest_folder = get_save_path(self.dest_folder, dset, ext=None)

            stim = {}
            if dset == 'allen':
                boc = BrainObservatoryCache(manifest_file='../data/brain_observatory_manifest.json')
                stim_types = [config['stim_types']] if isinstance(config['stim_types'], str) else config['stim_types']
                for stim_type in stim_types:
                    stim[dset, stim_type] = get_stimulus_allen(boc=boc, stim_type=stim_type, seq_len=seq_len)

            elif dset == 'majaj' or dset == 'movshon':
                images = pd.read_json(config['json_path_images'])
                if dset == 'majaj':
                    images, _ = image_order_majaj(images)
                stim[dset] = get_stimulus_monkey(images=images, root=config['img_root'])

            else:
                raise ValueError(f'Invalid dataset {dset}')

            for stim_type, stimulus in stim.items():
                print(f'Processing dataset {stim_type}')
                fname = f'{stim_type[1]}_{name}' if stim_type != dset else name
                self.load(dest_folder, fname, stim_type, 'rdms')
                if not force_activations and self.rdms[stim_type] is not None:
                    continue

                dl = get_stimulus_dl(data=stimulus, dset=dset, seq_len=seq_len, norm_kwargs=norm_kwargs)
                activations_pixel = self.get_activations_pixel(dl)
                activations_models = self.get_activations(dl, models)
                activations = [activations_pixel] + activations_models
                self.activations[stim_type] = activations

                if self.rdms[stim_type] is None:
                    print('Calculating rdms')
                    rdms = self.calculate_rdms(activations, dset)
                    self.rdms[stim_type] = rdms

                self.save(dest_folder, fname, rdms=self.rdms[stim_type])
        print()

    def get_activations(self, dl, models):
        activations = []

        for model in models:
            layers = model.layers
            acts = {n: [] for n in layers}
            model.cuda()

            for batch in dl:
                x = batch.cuda()
                with torch.no_grad():
                    out = model(x)
                for n in layers:
                    acts[n].append(out[n].detach().cpu())

            acts = {n: torch.cat(v, 0).mean(2) for n, v in acts.items()}
            activations.append(acts)
        return activations

    @staticmethod
    def get_activations_pixel(dl):
        """
        Get response matrix of the stimulus itself.

        Parameters
        ----------
        dl : DataLoader
            Stimulus DataLoader.

        Returns
        -------
        torch.Tensor
            Response matrix of the stimulus.
        """
        stim = []
        for batch in dl:
            stim.append(batch.detach().cpu())

        activations = torch.cat(stim, 0).mean((1, 2))
        return {"pixel": activations}

    @staticmethod
    def calculate_rdms(activations: List[dict], dset: str):
        rdms = []

        for act in activations:
            if dset == 'majaj':
                act = {k: v.unflatten(0, (-1, 50)).mean(1) for k, v in act.items()}
            act_dset = {k: rsa.data.Dataset(v.flatten(1).numpy()) for k, v in act.items()}
            rdm = {k: rsa.rdm.calc_rdm(v, method="correlation") for k, v in act_dset.items()}
            rdms.append(rdm)

        rdms = merge_dicts(rdms)
        rdms = {k: rsa.rdm.concat(v) for k, v in rdms.items()}
        return rdms


def load_models(ckpt_paths: List[str]):
    models = []
    norm_kwargs = None
    seq_len = None
    for ckpt_path in ckpt_paths:
        model, _norm_kwargs, _seq_len = load_model(ckpt_path)
        if norm_kwargs is None:
            norm_kwargs = _norm_kwargs
        if seq_len is None:
            seq_len = _seq_len

        assert seq_len == _seq_len, f'inconsistent seq_len {seq_len} in model {ckpt_path}'
        assert norm_kwargs == _norm_kwargs, f'inconsistent norm_kwargs {norm_kwargs} in model {ckpt_path}'

        models.append(model)
        norm_kwargs = _norm_kwargs
        seq_len = _seq_len
    return models, norm_kwargs, seq_len


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
    seq_len = model.network.last_duration
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


def get_stimulus_allen(boc, stim_type: str, seq_len: int):
    """
    Get stimulus data.

    Parameters
    ----------
    boc : BrainObservatoryCache
        BrainObservatoryCache object.
    stim_type : str
        Which stimulus type to use, can be one of 'natural_scenes', 'natural_movie_one', 'natural_movie_two',
        'natural_movie_three'.
    seq_len : int
        Sequence length to use for the stimulus.

    Returns
    -------
    np.ndarray
        Stimulus data.
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


def get_stimulus_monkey(images: pd.DataFrame, root: str):
    images = images.drop_duplicates(subset='Image_file', keep='first')

    data = []
    for img_path in images.Image_file:
        img_path = os.path.join(root, img_path)
        with open(img_path, "rb") as f:
            img = Image.open(f)
            data.append(np.array(img, np.uint8, copy=True))

    data = np.array(data)
    data = data[:, :, :, None] if len(data.shape) == 3 else data
    return data


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


def get_stimulus_dl(data: np.ndarray, dset: str, seq_len: int, norm_kwargs: Optional[Dict[str, Any]] = None):
    """
    Get stimulus dataset.

    Parameters
    ----------
    data : str
        The stimulus data.
    dset : str
        The dataset name.
    seq_len : int
        Sequence length to use for the stimulus.
    norm_kwargs : dict
        Normalization values (mean and std).

    Returns
    -------
    DataLoader
        Stimulus DataLoader.
    """
    transforms = T.Compose([
        T.ToTensor(),
        T.Resize((64, 64), antialias=True),
        T.Lambda(lambda x: x.unsqueeze(1) if len(x.shape) == 3 else x) if dset == 'majaj'
        else T.Lambda(lambda x: x.unsqueeze(0) if len(x.shape) == 3 else x),
        T.Lambda(lambda x: x.expand(3, seq_len, -1, -1)),
        TV.Normalize(**norm_kwargs) if norm_kwargs is not None else T.Lambda(lambda x: x),
    ])

    ds = StimuliDataset(data, transform=transforms)
    return DataLoader(ds, batch_size=256, num_workers=4, shuffle=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorchvideo.transforms as TV
import rsatoolbox as rsa
import seaborn as sns
import torch
import torchvision.transforms as T
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from dpc.models.dpc_plus_lit import DPCPlusLit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models._utils import IntermediateLayerGetter

boc = BrainObservatoryCache(
    manifest_file='../data/brain_observatory_manifest.json')


def plot_rsa(kts, save_path=None):
    data = []
    for species, kt in kts.items():
        df_model = pd.DataFrame(kt).melt(var_name='Layer', value_name='RSA')
        df_model['Model'] = species
        df_model['Path'] = [layer[:2] if layer[:2] in ['p1', 'p2'] else '' for layer in df_model['Layer']]
        df_model['Layer'] = [layer[3:] if layer[:2] in ['p1', 'p2'] else layer for layer in df_model['Layer']]
        data.append(df_model)

    df = pd.concat(data)
    df['Model_Path'] = df['Model'] + df['Path'].replace({'': '', 'p1': ' P1', 'p2': ' P2'})

    plt.figure(figsize=(10, 6))
    cs = sns.color_palette("Paired", n_colors=4)
    sns.pointplot(data=df, x='Layer', y='RSA', hue='Model_Path', palette=[cs[1], cs[1], cs[0], cs[3], cs[3], cs[2]],
                  dodge=0.4, errwidth=2, markersize=3, markers='o', capsize=0.1, join=False)
    plt.xlabel('Layers')
    plt.ylabel('RSA')
    plt.legend(title='Model')
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.title(f'RSA Comparison of SSL Models')
    plt.show()


def merge_dicts(dicts: list):
    super_dict = {}
    for d in dicts:
        for k, v in d.items():
            super_dict.setdefault(k, []).append(v)
    return super_dict


def cat_rdms(rdms: list):
    rdm_matrices = np.concatenate([rdm.get_matrices() for rdm in rdms], axis=0)
    return rsa.rdm.RDMs(rdm_matrices, "correlation")


def load_model(ckpt_path):
    model = DPCPlusLit.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    model.freeze()
    norm_kwargs = {"mean": model.hparams.mean, "std": model.hparams.std}
    seq_len = model.network.seq_len
    model = Model(model)
    return model, norm_kwargs, seq_len


def extract_one_path(model, path):
    m = nn.Sequential()
    for i, block in enumerate(model.network.backbone.get_submodule(f"path{path}").res_blocks.children()):
        m.add_module(f"p{path + 1}_res_block{i + 1}", block)
    m.add_module("out", model.network.backbone.get_submodule(f"dropout{path}"))

    layers = [n for n, _ in m.named_children()]
    m = IntermediateLayerGetter(m, dict(zip(layers, layers)))
    return m


def get_stimulus(stim_type, norm_kwargs, seq_len):
    transforms = T.Compose([
        T.ToTensor(), T.Resize((64, 64), antialias=True),
        T.Lambda(lambda x: x.expand(3, seq_len, -1, -1)),
        TV.Normalize(**norm_kwargs),
    ])
    data_set = boc.get_ophys_experiment_data(501498760)
    data = data_set.get_stimulus_template(stim_type)

    n, h, w = data.shape
    if stim_type == "natural_scenes":
        data = data[:, :, :, None]
    elif stim_type in ['natural_movie_one', 'natural_movie_two', 'natural_movie_three']:
        data = data.reshape(-1, seq_len, h, w).transpose(0, 2, 3, 1)
    else:
        raise ValueError(f'Invalid stimulus type {stim_type}')

    ds = StimuliDataset(data, transform=transforms)
    return DataLoader(ds, batch_size=32, num_workers=8, shuffle=False)


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

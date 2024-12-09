from typing import Optional

import rsatoolbox as rsa

from .rdms_model import ModelRDMs
from .rdms_monkey import MajajRDMs, MovshonRDMs
from .rdms_mouse import AllenRDMs
from .util import load_object, save_object, get_save_path


def calculate_rsa(model_config, dset_config, dest_folder: Optional[str] = None, num_iter: Optional[int] = None,
                  seed: Optional[int] = None):
    models = {}
    for model_name, ckpt_paths in model_config.items():
        models[model_name] = ModelRDMs(dset_config, ckpt_paths, model_name, dest_folder=dest_folder, seed=seed)

    brain_areas = {}
    for dset, dset_kwargs in dset_config.items():
        if dset == 'allen':
            for stim_type in dset_kwargs.pop('stim_types'):
                brain_areas[dset, stim_type] = AllenRDMs(**dset_kwargs, stim_types=stim_type,
                                                         dest_folder=dest_folder, num_iter=num_iter, seed=seed)
        elif dset == 'movshon':
            brain_areas[dset] = MovshonRDMs(**dset_kwargs, dest_folder=dest_folder, num_iter=num_iter, seed=seed)
        elif dset == 'majaj':
            brain_areas[dset] = MajajRDMs(**dset_kwargs, dest_folder=dest_folder, num_iter=num_iter, seed=seed)

    dest_folder = get_save_path(dest_folder, f'seed_{seed}', ext=None)
    sims = {}
    for dset_and_stim, brain_area in brain_areas.items():
        for model_name, model in models.items():
            model_rdms = model.rdms[dset_and_stim]
            for area, rdms in brain_area.rdms.items():
                if 'allen' in dset_and_stim:
                    name = '_'.join([dset_and_stim[1], *[str(v) for v in area]])
                    dset = 'allen'
                else:
                    name = area
                    dset = dset_and_stim

                print(f'Calculating RDM similarity between {model_name} and {dset}: {name}')
                path = get_save_path(dest_folder, dset, 'rsa', model_name, name, ext='pt')
                sim = load_object(path)
                if sim is not None:
                    sims[(dset_and_stim, model_name, area)] = sim
                    continue

                sim = {k: rsa.rdm.compare(rdms, rdm, method='tau-a').mean(0) for k, rdm in model_rdms.items()}
                sims[(model_name, name)] = sim
                save_object(sim, path)
    return sims

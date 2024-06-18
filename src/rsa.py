import rsatoolbox as rsa
import os

from src.rdms import get_rdms_allen, get_rdms_model
from src.util import plot_rsa


def compare_rsa(models, areas=[('VISp', 275, 'Cux2-CreERT2')], stim_type='natural_movie_one', num_iter=100,
                subsample=15, fig_path=None):
    all_kts = {}
    all_rdms = {}
    all_nc = {}
    for a in areas:
        area, depth, cre_line = a
        kts, rdms, noise_ceiling = calculate_rsa(area=area, depth=depth, cre_line=cre_line, stim_type=stim_type,
                                                 num_iter=num_iter, models=models, subsample=subsample)
        if fig_path is not None:
            out_dir = os.path.join(fig_path, area, stim_type)
            os.makedirs(out_dir, exist_ok=True)
            plot_rsa(kts, os.path.join(out_dir, f'{depth}_{cre_line}.svg'))
        all_kts[a] = kts
        all_rdms[a] = rdms
        all_nc[a] = noise_ceiling

    return all_kts, all_rdms, all_nc


def calculate_rsa(area, depth, cre_line, stim_type, num_iter, models, subsample=15):
    rdms_allen, noise_ceiling = get_rdms_allen(area=area, depth=depth, cre_line=cre_line, stim_type=stim_type,
                                               num_iter=num_iter, subsample=subsample)
    all_rdms = {"allen": rdms_allen}
    all_kts = {}
    for species, ckpt_paths in models.items():
        rdms = get_rdms_model(ckpt_paths, stim_type=stim_type)
        print(f"Calculating RSA for {species}")
        kts = {k: rsa.rdm.compare(rdms_allen, rdm, method='tau-a').mean(0) for k, rdm in rdms.items()}
        all_rdms[species] = rdms
        all_kts[species] = kts
        print("Done\n")

    return all_kts, all_rdms, noise_ceiling

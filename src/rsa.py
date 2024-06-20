import rsatoolbox as rsa
import os
import torch

from src.rdms import get_rdms_allen, get_rdms_model
from src.util import plot_rsa


def compare_rsa(models, areas=[('VISp', 275, 'Cux2-CreERT2')], stim_type='natural_movie_one', num_iter=100,
                seq_len=15, fig_path=None):
    all_rsas = {}
    all_rdms = {}
    all_nc = {}
    for a in areas:
        area, depth, cre_line = a

        path = os.path.join(fig_path if fig_path is not None else '', area, stim_type, f'{depth}_{cre_line}.pt')
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
                torch.save({'rsa': rsas, 'rdm': rdms, 'nc': noise_ceiling}, save_path + '.pt')

        plot_rsa(rsas, area=a, noise_ceiling=noise_ceiling, stim_type=stim_type, noise_corrected=False,
                 save_path=save_path)
        plot_rsa(rsas, area=a, noise_ceiling=noise_ceiling, stim_type=stim_type, noise_corrected=True,
                 save_path=save_path)

        all_rsas[a] = rsas
        all_rdms[a] = rdms
        all_nc[a] = noise_ceiling

    return all_rsas, all_rdms, all_nc


def calculate_rsa(area, depth, cre_line, stim_type, num_iter, models, seq_len):
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

import numpy as np
import rsatoolbox as rsa

from .activations import get_activations_model, get_activations_allen, get_activations_pixel
from .util import merge_dicts, cat_rdms


def get_rdms_model(ckpt_paths, stim_type, seq_len):
    print(f"Calculating RDMs for pixels")
    rdm_pixel = get_rdms_pixel(stim_type=stim_type, seq_len=seq_len)
    rdms = [rdm_pixel] * len(ckpt_paths)

    for ckpt_path in ckpt_paths:
        print(f"Calculating RDMs for {ckpt_path}")
        act = get_activations_model(ckpt_path=ckpt_path, stim_type=stim_type, seq_len=seq_len)
        act_dset = {k: rsa.data.Dataset(v.flatten(1).numpy()) for k, v in act.items()}
        rdm = {k: rsa.rdm.calc_rdm(v, method="correlation") for k, v in act_dset.items()}
        rdms.append(rdm)

    rdms = merge_dicts(rdms)
    rdms = {k: cat_rdms(v) for k, v in rdms.items()}
    return rdms


def get_rdms_pixel(stim_type, seq_len):
    act = get_activations_pixel(stim_type=stim_type, seq_len=seq_len)
    act_dset = rsa.data.Dataset(act.flatten(1).numpy())
    rdm = rsa.rdm.calc_rdm(act_dset, method="correlation")
    return {'pixel': rdm}


def get_rdms_allen(area, depth, cre_line, stim_type, num_iter, seq_len):
    print(f"Calculating RDMs for {stim_type} {area} {depth} {cre_line}")
    activations = get_activations_allen(area=area, depth=depth, cre_line=cre_line, stim_type=stim_type,
                                        seq_len=seq_len)
    num_trials = activations.shape[0]
    noise_ceiling = estimate_rdm_noise_ceiling(activations=activations,
                                               num_iter=num_iter if num_iter is not None else num_trials)

    rdms = []
    if num_iter is not None:
        for i in range(num_iter):
            trials_idx_permute = np.random.permutation(np.arange(0, num_trials))
            which_trials = trials_idx_permute[0:num_trials // 2]

            act = activations[which_trials].mean(0)
            act_dset = rsa.data.Dataset(act)
            rdm = rsa.rdm.calc_rdm(act_dset, method="correlation")
            rdms.append(rdm)
    else:
        for act in activations:
            act_dset = rsa.data.Dataset(act)
            rdm = rsa.rdm.calc_rdm(act_dset, method="correlation")
            rdms.append(rdm)

    rdms = cat_rdms(rdms)
    return rdms, noise_ceiling


def estimate_rdm_noise_ceiling(activations, num_iter):
    # activations are T (number of trials) x M (number of stimuli) x N (number of neurons)
    num_trials = activations.shape[0]
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

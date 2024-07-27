import numpy as np
import torch
import pandas as pd

from .util import boc, load_model, get_stimulus


def get_activations_model(ckpt_path: str, mode: str, seq_len: int, **stim_kwargs):
    """
    Get activations from a dual path model checkpoint.

    Parameters
    ----------
    ckpt_path : str
        Path to the model checkpoint.
    mode : str
        Which mode to use, can be one of 'allen', 'monkey'.
    seq_len : int
        Sequence length to use for the stimulus and model.
    stim_kwargs :
        Additional arguments for the stimulus dataset.

    Returns
    -------
    dict
        Dictionary of activations for each layer in the model.
    """
    print(f"Collecting activations for {ckpt_path}")
    model, norm_kwargs, seq_len_model = load_model(ckpt_path)
    assert seq_len == seq_len_model, f'seq_len {seq_len} does not match seq_len {seq_len_model} in model {ckpt_path}'
    dl = get_stimulus(mode=mode, seq_len=seq_len, norm_kwargs=norm_kwargs, **stim_kwargs)
    layers = model.layers

    activations = {n: [] for n in layers}
    model.cuda()
    for batch in dl:
        x = batch.cuda()
        with torch.no_grad():
            out = model(x)
        for n in layers:
            activations[n].append(out[n].detach().cpu())

    activations = {n: torch.cat(v, 0).mean(2) for n, v in activations.items()}
    return activations


def get_activations_pixel(mode: str, seq_len: int, **stim_kwargs):
    """
    Get response matrix of the stimulus itself.

    Parameters
    ----------
    mode : str
        Which mode to use, can be one of 'allen', 'monkey'.
    seq_len : int
        Sequence length to use for the stimulus.
    stim_kwargs :
        Additional arguments for the stimulus dataset.

    Returns
    -------
    torch.Tensor
        Response matrix of the stimulus.
    """
    dl = get_stimulus(mode=mode, seq_len=seq_len, norm_kwargs=None, **stim_kwargs)
    stim = []
    for batch in dl:
        stim.append(batch.detach().cpu())
    return torch.cat(stim, 0).mean((1, 2))


def get_activations_allen(area: str, depth: int, cre_line: str, stim_type: str, seq_len: int, verbose: bool = False):
    """
    Get mouse brain activations from the Allen Brain Observatory.

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
    seq_len : int
        Sequence length to use for the stimulus.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    np.ndarray
        Activations for the given parameters.
    """
    print(f"Collecting activations")
    all_activations = []
    all_ecs = boc.get_experiment_containers(cre_lines=[cre_line], targeted_structures=[area], imaging_depths=[depth])
    if verbose:
        print(f"number of {cre_line} experiment containers: {len(all_ecs)}")

    for ecs in all_ecs:
        exps = boc.get_ophys_experiments(experiment_container_ids=[ecs['id']], stimuli=[stim_type])
        if verbose:
            print(f"experiment container: {ecs['id']}; num experiments: {len(exps)}")

        for exp in exps:
            data_set = boc.get_ophys_experiment_data(exp['id'])
            events = boc.get_ophys_experiment_events(exp['id'])

            if stim_type == 'natural_scenes':
                activations = get_activations_ns(data_set=data_set, event=events, verbose=verbose)
            elif stim_type in ['natural_movie_one', 'natural_movie_two', 'natural_movie_three']:
                activations = get_activations_nm(data_set=data_set, event=events, movie=stim_type, seq_len=seq_len,
                                                 verbose=verbose)
            else:
                raise ValueError(f'Invalid stimulus type {stim_type}')

            all_activations.append(activations)
    return np.concatenate(all_activations, axis=2)


def get_activations_ns(data_set, event: np.ndarray, verbose: bool = False):
    """
    Get activations for natural scenes stimulus.

    Parameters
    ----------
    data_set : BrainObservatoryNwbDataSet
        BrainObservatoryNwbDataSet object which contains the data.
    event : np.ndarray
        Event data for the session.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    np.ndarray
        Activations for the given parameters.
    """
    stim_table = data_set.get_stimulus_table('natural_scenes')
    all_cell_ids = data_set.get_cell_specimen_ids()

    num_trials = 50
    num_images = 118
    num_neurons = len(all_cell_ids)
    if verbose:
        print('there are ' + str(num_neurons) + ' neurons in this session')

    activations = np.empty([num_trials, num_images, num_neurons])

    for neuron, cell in enumerate(event):
        for img in range(num_images):
            this_stim = stim_table[(stim_table.frame == img)].to_numpy()
            for trial in range(this_stim.shape[0]):
                start = int(this_stim[trial, 1])
                end = start + 15
                activations[trial, img, neuron] = np.sum(cell[start:end])

    return activations


def get_activations_nm(data_set, event: np.ndarray, movie: str, seq_len: int, verbose: bool = False):
    """
    Get activations for natural scenes stimulus.

    Parameters
    ----------
    data_set : BrainObservatoryNwbDataSet
        BrainObservatoryNwbDataSet object which contains the data.
    event : np.ndarray
        Event data for the session.
    movie : str
        Which movie to use, can be one of 'natural_movie_one', 'natural_movie_two', 'natural_movie_three'.
    seq_len : int
        Sequence length to use for the stimulus.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    np.ndarray
        Activations for the given parameters.
    """
    stim_table = data_set.get_stimulus_table(movie)
    all_cell_ids = data_set.get_cell_specimen_ids()
    num_neurons = len(all_cell_ids)
    if verbose:
        print('there are ' + str(num_neurons) + ' neurons in this session')

    movie_len = len(stim_table[stim_table.repeat == 0])
    num_trials = 10
    activations = np.empty([num_trials, movie_len, num_neurons])

    for neuron, cell in enumerate(event):
        for trial in range(max(stim_table.repeat) + 1):
            start = stim_table.start[stim_table.repeat == trial] + 0
            activations[trial, :, neuron] = cell[start]

    if seq_len > 1:
        assert movie_len % seq_len == 0, f'movie length {movie_len} must be divisible by seq_len {seq_len}'
        activations = activations.reshape(num_trials, movie_len // seq_len, seq_len, num_neurons).mean(2)

    return activations


def get_activations_monkey(json_path: str, area: str):
    """
    Get monkey brain activations.

    Parameters
    ----------
    json_path : str
        Path to the json file containing the responses.
    area : str
        Which brain area to use. Can be one of 'V1', 'V2'.
    """
    assert area in ['V1', 'V2'], f'Invalid area {area}'
    print(f"Collecting activations")
    num_trials = 20
    num_images = 135

    responses = pd.read_json(json_path)
    responses['Value'] = responses['Value'].apply(lambda x: np.array(x).reshape(300, -1)[50:250].sum(axis=0))
    responses = responses[responses.Region == area]
    activations = np.array(responses.Value.tolist()).reshape(-1, num_images, num_trials).T
    return activations

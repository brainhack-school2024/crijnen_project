from typing import Optional, Union, List

import numpy as np
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

from .rdms import BrainRDMs
from .util import get_save_path


class AllenRDMs(BrainRDMs):
    def __init__(
            self,
            stim_types: Union[List[str], str],
            areas: Union[List[str], str],
            depths: Union[List[int], int],
            cre_lines: Union[List[str], str],
            seq_len: Optional[int] = 1,
            data_folder: Optional[str] = '../data/brain_observatory_manifest.json',
            dest_folder: Optional[str] = None,
            force_activations: Optional[bool] = False,
            num_iter: Optional[int] = None,
            seed: Optional[int] = None,
            verbose: Optional[bool] = False,
            **kwargs
    ):
        """
        Calculate RDMs for mouse brain data from the Allen Brain Observatory.

        Parameters
        ----------
        stim_types : Union[List[str], str]
            Which stimuli to use, can be one or multiple of 'natural_scenes', 'natural_movie_one',
            'natural_movie_two', 'natural_movie_three'.
        areas : Union[List[str], str]
            Which brain areas to use. Can be one or multiple of 'VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl'.
        depths : Union[List[int], int]
            Which depths to use.
        cre_lines : Union[List[str], str]
            Which cre lines to use.
        seq_len : int
            Used for downsampling the natural movie stimuli. The number of frames to average the response over.
            If seq_len is 1, it will not downsample the data.
        data_folder : str
            Path to the data folder. This is where the Allen Brain Observatory data will be downloaded and stored.
        dest_folder : Optional[str]
            Path to the destination folder. If dest_folder is not None, it will try to load the Noise Ceilings and RDMs
            from the folder and if they don't exist it will save them to the folder.
        force_activations : Optional[bool]
            If force_activations is True, it will recalculate the activations after loading the Noise Ceilings and RDMs.
        num_iter : Optional[int]
            Number of iterations to use for bootstrapping.
        seed : Optional[int]
            Random seed.
        verbose : Optional[bool]
            Whether to print verbose when processing the data.
        """
        super().__init__(dest_folder=dest_folder, seed=seed)
        print('Processing Allen Brain Observatory data')

        self.stim_types = [stim_types] if isinstance(stim_types, str) else stim_types
        self.areas = [areas] if isinstance(areas, str) else areas
        self.depths = [depths] if isinstance(depths, int) else depths
        self.cre_lines = [cre_lines] if isinstance(cre_lines, str) else cre_lines
        self.boc = BrainObservatoryCache(manifest_file=data_folder)
        self.dest_folder = get_save_path(self.dest_folder, 'allen', ext=None)

        for stim_type in self.stim_types:
            for area in self.areas:
                for depth in self.depths:
                    for cre_line in self.cre_lines:
                        print(f'Processing stim_type: {stim_type}, area: {area}, depth: {depth}, cre_line: {cre_line}')
                        key = (stim_type, area, depth, cre_line)
                        fname = f'{stim_type}_{area}_{depth}_{cre_line}'
                        self.load(self.dest_folder, fname, key, 'noise_ceilings', 'rdms')
                        if not force_activations and (self.noise_ceilings[key] is not None
                                                      and self.rdms[key] is not None):
                            continue

                        activations = self.get_activations(stim_type=stim_type, area=area, depth=depth,
                                                           cre_line=cre_line, seq_len=seq_len, verbose=verbose)
                        self.activations[key] = activations

                        if self.noise_ceilings[key] is None:
                            print('Calculating noise ceiling')
                            noise_ceiling = self.estimate_noise_ceiling(activations, num_iter=num_iter, seed=seed)
                            self.noise_ceilings[key] = noise_ceiling

                        if self.rdms[key] is None:
                            print('Calculating rdms')
                            rdms = self.calculate_rdms(activations, num_iter=num_iter, seed=seed)
                            self.rdms[key] = rdms

                        self.save(self.dest_folder, fname, noise_ceilings=self.noise_ceilings[key], rdms=self.rdms[key])
        print()

    def get_activations(self, stim_type: str, area: str, depth: int, cre_line: str, seq_len: int,
                        verbose: bool = False):
        """
        Get mouse brain activations from the Allen Brain Observatory.

        Parameters
        ----------
        stim_type : str
            Which stimulus type to use, can be one of 'natural_scenes', 'natural_movie_one', 'natural_movie_two',
            'natural_movie_three'.
        area : str
            Which brain area to use. Can be one of 'VISp', 'VISl', 'VISal', 'VISpm', 'VISam', 'VISrl'.
        depth : int
            Which depth to use.
        cre_line : str
            Which cre line to use.
        seq_len : int
            Sequence length to use for the stimulus.
        verbose : bool
            Whether to print verbose output.

        Returns
        -------
        np.ndarray
            Activations for the given parameters.
        """
        all_activations = []
        all_ecs = self.boc.get_experiment_containers(cre_lines=[cre_line], targeted_structures=[area],
                                                     imaging_depths=[depth])
        if verbose:
            print(f"number of {cre_line} experiment containers: {len(all_ecs)}")

        for ecs in all_ecs:
            exps = self.boc.get_ophys_experiments(experiment_container_ids=[ecs['id']], stimuli=[stim_type])
            if verbose:
                print(f"experiment container: {ecs['id']}; num experiments: {len(exps)}")

            for exp in exps:
                data_set = self.boc.get_ophys_experiment_data(exp['id'])
                events = self.boc.get_ophys_experiment_events(exp['id'])

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
        The number of frames to average the response over.
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

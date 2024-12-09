from typing import Optional, Union, List

import numpy as np
import pandas as pd

from .rdms import BrainRDMs
from .util import get_save_path


class MonkeyRDMs(BrainRDMs):
    dset = None

    def __init__(
            self,
            json_path_responses,
            json_path_images,
            areas: Union[List[str], str],
            dest_folder: Optional[str] = None,
            force_activations: Optional[bool] = False,
            num_iter: Optional[int] = None,
            seed: Optional[int] = None,
            **kwargs
    ):
        super().__init__(dest_folder=dest_folder, seed=seed)
        print(f'Processing {self.dset} data')
        responses = pd.read_json(json_path_responses)
        images = pd.read_json(json_path_images)
        self.responses, self.images = self.prepare_data(responses, images)
        self.areas = [areas] if isinstance(areas, str) else areas
        self.dest_folder = get_save_path(self.dest_folder, self.dset, ext=None)
        self.num_iter = num_iter

        for area in self.areas:
            print(f'Processing area {area}')
            self.load(self.dest_folder, area, area, 'noise_ceilings', 'rdms')
            if not force_activations and (self.noise_ceilings[area] is not None and self.rdms[area] is not None):
                continue

            activations = self.get_activations(area)
            self.activations[area] = activations

            if self.noise_ceilings[area] is None:
                print('Calculating noise ceiling')
                noise_ceiling = self.estimate_noise_ceiling(activations, num_iter=num_iter, seed=seed)
                self.noise_ceilings[area] = noise_ceiling

            if self.rdms[area] is None:
                print('Calculating rdms')
                rdms = self.calculate_rdms(activations, num_iter=num_iter, seed=seed)
                self.rdms[area] = rdms

            self.save(self.dest_folder, area, noise_ceilings=self.noise_ceilings[area], rdms=self.rdms[area])
        print()

    @staticmethod
    def prepare_data(responses, images):
        raise NotImplementedError


class MovshonRDMs(MonkeyRDMs):
    dset = 'movshon'

    @staticmethod
    def prepare_data(responses, images):
        responses = prepare_data_movshon(responses)
        return responses, images

    def get_activations(self, area: str):
        assert area in ['V1', 'V2'], f'Invalid area {area}'
        responses = self.responses

        num_trials = 20
        num_images = 135

        responses = responses[responses.Region == area]
        activations = np.array(responses.Value.tolist()).reshape(-1, num_images, num_trials).T
        return activations


class MajajRDMs(MonkeyRDMs):
    dset = 'majaj'

    @staticmethod
    def prepare_data(responses, images):
        images, indices = image_order_majaj(images)
        responses = prepare_data_majaj(responses, indices)
        return responses, images

    def get_activations(self, area: str):
        assert area == 'V4', f'Invalid area {area}'
        responses = self.responses

        num_trials = 28
        num_categories = 64
        num_reps = 50

        responses = responses[responses.Region == area]
        activations = np.array(responses.NewValue.tolist()).reshape(-1, num_categories, num_reps, num_trials).mean(-2).T
        return activations


def prepare_data_movshon(responses):
    responses['Value'] = responses['Value'].apply(lambda x: np.array(x).reshape(300, -1)[50:250].sum(axis=0))
    return responses


def image_order_majaj(images):
    desired_order = ['Animals', 'Boats', 'Cars', 'Chairs', 'Faces', 'Fruits', 'Planes', 'Tables']
    images['Category'] = pd.Categorical(images['Category'], categories=desired_order, ordered=True)
    images = images.sort_values(['Category', 'Image_file'])

    grouped = images.groupby(['Category', 'Image_file']).head(28)
    indices = grouped.index

    images = images.loc[indices]
    return images, indices


def prepare_data_majaj(responses, indices):
    responses['NewValue'] = responses.Value.apply(lambda x: np.array(x)[indices])
    return responses

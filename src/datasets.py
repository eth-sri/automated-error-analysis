"""
- this file is used to load the ImageNet dataset
- it is based on the publicly available code
https://github.com/locuslab/smoothing/blob/master/code/datasets.py written by Jeremy Cohen and
https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/datasets.py written by Hadi Salman
"""

import os
import json
from typing import Any, Tuple

from torchvision import transforms, datasets

from src import utils


# set this environment variable to the location of your imagenet directory if you want to read ImageNet data, e.g.
# export IMAGENET_DIR=/home/.../imagenet
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"
IMAGENET_A_LOC_ENV = "IMAGENET_A_DIR"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

with open(utils.get_artefacts_path() / 'imagenet_wnet_id_to_target.json', 'r') as class_to_idx_file:
    wnet_id_to_target = json.load(class_to_idx_file)


def standard_transform(img_resize_size, img_crop_size, normalize=True):
    transforms_list = [
        transforms.Resize(img_resize_size),
        transforms.CenterCrop(img_crop_size),
        transforms.ToTensor()
    ]
    if normalize:
        transforms_list += [
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STDDEV)
        ]
    return transforms.Compose(transforms_list)


STANDARD_TRANSFORM = standard_transform(256, 224)


class ImageNetDataset(datasets.ImageFolder):

    def __init__(self, root, transform, dataset_id = 'imagenet'):
        super().__init__(root, transform)
        self.dataset_id = dataset_id

    def get_indices_in_1k(self):
        classes = self.classes
        indices_in_1k = [wnet_id_to_target[c] for c in classes]
        for i1, i2 in zip(indices_in_1k[:-1], indices_in_1k[1:]):
            assert i1 < i2
        return indices_in_1k

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (path, sample, target) where target is class_index of the target class.
        """
        path, default_target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        assert self.target_transform is None

        wnet_id = path.split('/')[-2]
        assert wnet_id in wnet_id_to_target
        if self.dataset_id == 'imagenet':
            assert default_target == wnet_id_to_target[wnet_id]

        return path, sample, wnet_id_to_target[wnet_id], default_target


def get_dataset_root(dataset: str = 'imagenet', split: str = 'val'):
    assert dataset in ['imagenet', 'imagenet-a']
    if dataset == 'imagenet':
        if IMAGENET_LOC_ENV not in os.environ:
            raise RuntimeError("Environment variable for ImageNet directory not set")
        imagenet_dir = os.environ[IMAGENET_LOC_ENV]
        return os.path.join(imagenet_dir, split)
    elif dataset == 'imagenet-a':
        assert split == 'val'
        if IMAGENET_A_LOC_ENV not in os.environ:
            raise RuntimeError("Environment variable for ImageNet-A directory not set")
        imagenet_a_dir = os.environ[IMAGENET_A_LOC_ENV]
        return imagenet_a_dir
    else:
        raise ValueError(f'Dataset {dataset} is not supported')


def get_dataset(dataset: str = 'imagenet', split: str = 'val', transform=STANDARD_TRANSFORM) -> ImageNetDataset:
    """Return the dataset as a PyTorch Dataset object"""
    root = get_dataset_root(dataset=dataset, split=split)
    return ImageNetDataset(root, transform, dataset_id=dataset)


if __name__ == '__main__':
    pass

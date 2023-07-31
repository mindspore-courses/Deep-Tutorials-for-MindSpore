# pylint: disable=C0103
# pylint: disable=E0401

"""CaptionDataset"""
import json
import os
import h5py
import mindspore
from mindspore import Tensor

class CaptionDataset:
    """
    A MindSpore Dataset class to be used in a MindSpore DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r', encoding='utf-8') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r', encoding='utf-8') as j:
            self.caplens = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        img = Tensor(self.imgs[i // self.cpi] / 255., dtype=mindspore.float32)

        caption = Tensor(self.captions[i], dtype=mindspore.int64)
        caplen = Tensor(self.caplens[i], dtype=mindspore.int64)
        if self.split == 'TRAIN':
            return img, caption, caplen
        # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
        all_captions = Tensor(
            self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)], dtype=mindspore.int64)
        return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size

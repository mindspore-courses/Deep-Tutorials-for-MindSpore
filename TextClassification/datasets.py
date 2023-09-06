# pylint: disable=C0103
"""HANDataset"""

import os
import pickle
import numpy as np

class HANDataset:
    """
    A Dataset class to be used in a MindSpore GeneratorDataset
    """
    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        split = split.upper()
        assert split in {'TRAIN', 'TEST'}
        self.split = split

        # Load data
        with open(os.path.join(data_folder, split + '_data.pkl'), 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, i):
        return np.array(self.data['docs'][i], dtype=np.float32), \
               np.array([self.data['sentences_per_document'][i]], dtype=np.float32), \
               np.array(self.data['words_per_sentence'][i], dtype=np.float32), \
               np.array([self.data['labels'][i]], dtype=np.float32)

    def __len__(self):
        return len(self.data['labels'])

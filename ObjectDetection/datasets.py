# pylint: disable=C0103
# pylint: disable=E0401
"""PascalVOCDataset"""

import json
import os
import numpy as np
from PIL import Image
from utils import transform

class PascalVOCDataset:
    """
    Pascal VOC Dataset
    """
    def __init__(self, data_folder, split, keep_difficult=False):
        self.split = split.uppper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r', encoding='utf-8') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r', encoding='utf-8') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):

        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        objects = self.objects[i]
        boxes = np.array(objects['boxes'], dtype=np.float32)
        labels = np.array(objects['labels'], dtype=np.float32)
        difficulties = np.array(objects['difficulties'], dtype=np.byte)

        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """collate fn"""
        images = []
        boxes = []
        labels = []
        difficulties = []

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = np.stack(images, axis=0)

        return images, boxes, labels, difficulties

# pylint: disable=C0103
"""utils"""

import os
import json
import random
import xml.etree.ElementTree as ET
import numpy as np
import mindspore
import mindspore.dataset.vision.transforms as VT
from mindspore import ops
from PIL import Image

# Label map
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def parse_annotation(annotation_path):
    """parse annotation"""
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    labels = []
    difficulties = []
    for obj in root.iter('object'):

        difficult = int(obj.find('difficult').text == '1')

        label = obj.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

def create_data_lists(voc07_path, voc12_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    """
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    train_images = []
    train_objects = []
    n_objects = 0

    # Training data
    for path in [voc07_path, voc12_path]:

        # Find IDs of images in training data
        with open(os.path.join(path, 'ImageSets/Main/trainval.txt'), encoding='utf-8') as f:
            ids = f.read().splitlines()

        for id_ in ids:
            # Parse annotation's XML file
            objects = parse_annotation(os.path.join(path, 'Annotations', id_ + '.xml'))
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
            train_images.append(os.path.join(path, 'JPEGImages', id_ + '.jpg'))

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w', encoding='utf-8') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w', encoding='utf-8') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w', encoding='utf-8') as j:
        json.dump(label_map, j)  # save label map too

    print(f'\nThere are {len(train_images)} training images containing a total of %d objects. Files have been saved to %s.' % (
         n_objects, os.path.abspath(output_folder)))

    # Test data
    test_images = []
    test_objects = []
    n_objects = 0

    # Find IDs of images in the test data
    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt'), encoding='utf-8') as f:
        ids = f.read().splitlines()

    for id_ in ids:
        # Parse annotation's XML file
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id_ + '.xml'))
        if len(objects) == 0:
            continue
        test_objects.append(objects)
        n_objects += len(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id_ + '.jpg'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w', encoding='utf-8') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w', encoding='utf-8') as j:
        json.dump(test_objects, j)

    print(f'\nThere are {len(test_images)} test images containing a total of {n_objects} \
          objects. Files have been saved to {os.path.abspath(output_folder)}.')

def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(axis=d, index=ops.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor

def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = []
    for i, true_label in enumerate(true_labels):
        true_images.extend([i] * true_label.shape[0])
    true_images = mindspore.Tensor(true_images, mindspore.int64)
    true_boxes = ops.cat(true_boxes, axis=0)  # (n_objects, 4)
    true_labels = ops.cat(true_labels, axis=0)  # (n_objects)
    true_difficulties = ops.cat(true_difficulties, axis=0)  # (n_objects)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = []
    for i, det_label in enumerate(det_labels):
        det_images.extend([i] * det_label.shape[0])
    det_images = mindspore.Tensor(det_images, mindspore.int64)  # (n_detections)
    det_boxes = ops.cat(det_boxes, axis=0)  # (n_detections, 4)
    det_labels = ops.cat(det_labels, axis=0)  # (n_detections)
    det_scores = ops.cat(det_scores, axis=0)  # (n_detections)

    # Calculate APs for each class (except background)
    average_precisions = ops.zeros((n_classes - 1), dtype=mindspore.float32)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = float((1 - true_class_difficulties).sum())  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = ops.zeros((true_class_difficulties.shape[0]), dtype=mindspore.uint8)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.shape[0]
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = ops.sort(det_class_scores, axis=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = ops.zeros((n_class_detections), dtype=mindspore.float32)  # (n_class_detections)
        false_positives = ops.zeros((n_class_detections), dtype=mindspore.float32)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.shape[0] == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = ops.max(overlaps.squeeze(0), axis=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = mindspore.Tensor(range(true_class_boxes.shape[0]))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if float(max_overlap) > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = ops.cumsum(true_positives, axis=0)  # (n_class_detections)
        cumul_false_positives = ops.cumsum(false_positives, axis=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = ops.arange(start=0, end=1.1, step=.1).asnumpy().tolist()  # (11)
        precisions = ops.zeros((len(recall_thresholds)), dtype=mindspore.float32)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = float(average_precisions.mean())

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.asnumpy().tolist())}

    return average_precisions, mean_average_precision

def xy_to_cxcy(xy):
    """xy to cxcy"""
    return ops.cat([(xy[:, 2:] + xy[:, :2]) / 2,
                    xy[:, 2:] - xy[:, :2]], 1)

def cxcy_to_xy(cxcy):
    """cxcy to xy"""
    return ops.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),
                    cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)

def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """cxcy to gcxgcy"""
    return ops.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),
                    ops.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)

def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """gcxgcy to cxcy"""
    return ops.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                    ops.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)

def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    lower_bounds = ops.maximum(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = ops.minimum(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = ops.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]

def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.
    """

    original_h = image.shape[1]
    original_w = image.shape[2]
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    filler = mindspore.Tensor(filler, mindspore.int64)
    new_image = ops.ones((3, new_h, new_w), dtype=mindspore.float32) * filler.unsqueeze(1).unsqueeze(1)

    left =  random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    new_boxes = boxes + mindspore.Tensor([left, top, left, top], dtype=mindspore.float32).unsqueeze(0)

    return new_image, new_boxes

def random_crop(image, boxes, labels, difficulties):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = mindspore.Tensor([left, top, right, bottom], mindspore.float32)  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes)  # (1, n_objects), n_objects is the no. of objects in this image

            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if float(overlap.max()) < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = ops.maximum(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = ops.minimum(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties

def flip(image, boxes):
    """
    Flip image horizontally.
    """
    hflip = VT.RandomHorizontalFlip(prob=1)
    new_image = hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes

def resize(image, boxes, size=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).
    """
    new_image = VT.Resize(size)(image)

    old_size = mindspore.Tensor([image.width, image.height, image.width, image.height], dtype=mindspore.float32).unsqueeze(0)
    new_boxes = boxes / old_size

    if not return_percent_coords:
        new_size = mindspore.Tensor([size[1], size[0], size[1], size[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_size

    return new_image, new_boxes

def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    """

    distortions = [VT.AdjustBrightness,
                   VT.AdjustContrast,
                   VT.AdjustSaturation,
                   VT.AdjustHue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            new_image = d(adjust_factor)(image)

    return new_image

def transform(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties

    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = VT.ToTensor()(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = Image.fromarray(np.uint8(255*new_image.transpose(1, 2, 0)))

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, size=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = VT.ToTensor()(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = VT.Normalize(mean=mean, std=std)(new_image.transpose(1, 2, 0))

    return new_image, new_boxes, new_labels, new_difficulties

def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.shape[0]
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.equal(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return float(correct_total) * (100.0 / batch_size)

class AverageMeter:
    """AverageMeter"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def clip_gradient(grad, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    """
    grad = ops.clamp(grad, -grad_clip, grad_clip)
    return grad

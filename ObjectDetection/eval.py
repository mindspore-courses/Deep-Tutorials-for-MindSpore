# pylint: disable=C0103
"""evaluate the model"""

from pprint import PrettyPrinter
from mindspore import load_checkpoint, load_param_into_net
from mindspore.dataset import GeneratorDataset
from utils import label_map, calculate_mAP
from model import SSD300
from datasets import PascalVOCDataset
from tqdm import tqdm

pp = PrettyPrinter()

data_folder = './'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64

checkpoint = ''
model = SSD300(n_classes=len(label_map))
params_dict = load_checkpoint(checkpoint)
load_param_into_net(model, params_dict)

dataset_source = PascalVOCDataset(data_folder, 'test', keep_difficult)
test_dataset = GeneratorDataset(dataset_source, column_names=['image', 'boxes', 'labels', 'difficulties'])
test_dataset = test_dataset.batch(batch_size=batch_size, per_batch_map=dataset_source.collate_fn)

def evaluate():
    """
    Evaluate
    """
    model.set_train(False)
    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []
    true_difficulties = []

    for images, boxes, labels, difficulties in tqdm(test_dataset.create_tuple_iterator(), desc='Evaluating'):

        # Forward prop.
        predicted_locs, predicted_scores = model(images)

        # Detect objects in SSD output
        det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                    min_score=0.01, max_overlap=0.45,
                                                                                    top_k=200)
        # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

        det_boxes.extend(det_boxes_batch)
        det_labels.extend(det_labels_batch)
        det_scores.extend(det_scores_batch)
        true_boxes.extend(boxes)
        true_labels.extend(labels)
        true_difficulties.extend(difficulties)

    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    pp.pprint(APs)

    print(f'\nMean Average Precision (mAP): {mAP:.3f}')

evaluate()

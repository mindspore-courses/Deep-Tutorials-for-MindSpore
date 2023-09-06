# pylint: disable=C0103
"""train the model"""

import time
import mindspore
from mindspore import nn
from mindspore.dataset import GeneratorDataset
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import label_map, clip_gradient, AverageMeter

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding


start_epoch = 0
model = SSD300(n_classes=n_classes)
# Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
biases = []
not_biases = []
for param, param_name in model.parameters_and_names():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
        else:
            not_biases.append(param)
optimizer = nn.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                            learning_rate=lr, momentum=momentum, weight_decay=weight_decay)

criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)
dataset_source = PascalVOCDataset(data_folder, 'train', keep_difficult)
train_dataset = GeneratorDataset(source=dataset_source,
                                    column_names=['image', 'boxes', 'labels', 'difficulties'])
train_dataset = train_dataset.batch(batch_size=batch_size, per_batch_map=dataset_source.collate_fn)

epochs = iterations // (len(train_dataset) // 32)
decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

def forward_fn(images, boxes, labels):
    """
    forward function
    """
    predicted_locs, predicted_scores = model(images)
    loss = criterion(predicted_locs, predicted_scores, boxes, labels)
    return loss

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameter, False)

def train_step(images, boxes, labels):
    """train step"""
    loss, grad = grad_fn(images, boxes, labels)

    if grad_clip is not None:
        grad = clip_gradient(grad, grad_clip)

    optimizer(grad)
    return loss

def train(epoch):
    """
    train
    """
    model.set_train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    for i, (images, boxes, labels, _) in enumerate(train_dataset.create_dict_iterator()):
        data_time.update(time.time() - start)

        loss = train_step(images, boxes, labels)

        losses.update(float(loss), images.shape[0])
        batch_time.update(time.time(), start)

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_dataset)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')

def main():
    """main"""
    for epoch in range(start_epoch, start_epoch + epochs):
        train(epoch)

if __name__ == '__main__':
    main()

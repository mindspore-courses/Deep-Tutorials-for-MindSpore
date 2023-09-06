# pylint: disable=C0103
"""train SRResNet"""

import time
import mindspore
from mindspore import nn
from mindspore import save_checkpoint
from mindspore.dataset import GeneratorDataset
from model import SRResNet
from datasets import SRDataset
from utils import clip_gradient, AverageMeter

data_folder = './'  # folder with JSON data files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Model parameters
large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks = 16  # number of residual blocks

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 16  # batch size
start_epoch = 0  # start at this epoch
iterations = 1e6  # number of training iterations
workers = 4
print_freq = 500  # print training status once every __ batches
lr = 1e-4  # learning rate
grad_clip = None  # clip if gradients are exploding

model = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                         n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)
        # Initialize the optimizer
optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

criterion = nn.MSELoss()
train_dataset_source = SRDataset(data_folder,
                                 split='train',
                                 crop_size=crop_size,
                                 scaling_factor=scaling_factor,
                                 lr_img_type='imagenet-norm',
                                 hr_img_type='[-1, 1]')
train_dataset = GeneratorDataset(train_dataset_source, column_names=['lr_img', 'hr_img'])
train_dataset = train_dataset.batch(batch_size=batch_size, num_parallel_workers=workers)

epochs = int(iterations // len(train_dataset) + 1)

def forward_fn(low_imgs, high_imgs):
    """forward fn"""
    super_imgs = model(low_imgs)
    loss = criterion(super_imgs, high_imgs)
    return loss

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameter)

def train_step(low_imgs, high_imgs):
    """train step"""
    grad, loss = grad_fn(low_imgs, high_imgs)

    if grad_clip is not None:
        grad = clip_gradient(grad, grad_clip)
    optimizer(grad)

    return loss

def train(epoch):
    """train"""
    model.set_train()
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    for i, (lr_imgs, hr_imgs) in enumerate(train_dataset.create_tuple_iterator()):
        data_time.update(time.time() - start)

        loss = train_step(lr_imgs, hr_imgs)
        losses.update(float(loss), lr_imgs.shape[0])
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_dataset)}]----'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')

def main():
    """main"""
    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
    save_checkpoint(model, f'srresnet_epoch_{epoch}.ckpt')

if __name__ == "__main__":
    main()

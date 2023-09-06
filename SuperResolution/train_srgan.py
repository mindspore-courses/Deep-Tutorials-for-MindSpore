# pylint: disable=C0103
"""train Generator and Discriminator in SRGAN"""

import time
import mindspore
from mindspore import nn, ops, save_checkpoint
from mindspore.dataset import GeneratorDataset
from datasets import SRDataset
from model import Generator, Discriminator, TruncatedVGG19
from utils import convert_image, clip_gradient, AverageMeter

# Data parameters
data_folder = './'  # folder with JSON data files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor

# Generator parameters
large_kernel_size_g = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size_g = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels_g = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks_g = 16  # number of residual blocks
srresnet_checkpoint = "./srresnet_epoch_0.ckpt"  # filepath of the trained SRResNet checkpoint used for initialization

# Discriminator parameters
kernel_size_d = 3  # kernel size in all convolutional blocks
n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
n_blocks_d = 8  # number of convolutional blocks
fc_size_d = 1024  # size of the first fully connected layer

# Learning parameters
checkpoint = None  # path to model (SRGAN) checkpoint, None if none
batch_size = 16  # batch size
start_epoch = 0  # start at this epoch
iterations = 2e5  # number of training iterations
workers = 4  # number of workers
vgg19_i = 5  # the index i in the definition for VGG loss; see paper or models.py
vgg19_j = 4  # the index j in the definition for VGG loss; see paper or models.py
beta = 1e-3  # the coefficient to weight the adversarial loss in the perceptual loss
print_freq = 500  # print training status once every __ batches
lr = 1e-4  # learning rate
grad_clip = None  # clip if gradients are exploding

generator = Generator(large_kernel_size=large_kernel_size_g,
                      small_kernel_size=small_kernel_size_g,
                      n_channels=n_channels_g,
                      n_blocks=n_blocks_g,
                      scaling_factor=scaling_factor)

# Initialize generator network with pretrained SRResNet
generator.initialize_with_srresnet(srresnet_checkpoint=srresnet_checkpoint)

# Initialize generator's optimizer
optimizer_g = nn.Adam(params=generator.trainable_params(), learning_rate=lr)

train_dataset_source = SRDataset(data_folder,
                                 split='train',
                                 crop_size=crop_size,
                                 scaling_factor=scaling_factor,
                                 lr_img_type='imagenet-norm',
                                 hr_img_type='imagenet-norm')
train_dataset = GeneratorDataset(train_dataset_source, ['lr_img', 'hr_img'])
train_dataset = train_dataset.batch(batch_size, num_parallel_workers=workers)

epochs = int(iterations // len(train_dataset) + 1)
# Discriminator
discriminator = Discriminator(kernel_size=kernel_size_d,
                                n_channels=n_channels_d,
                                n_blocks=n_blocks_d,
                                fc_size=fc_size_d)
optimizer_d = nn.Adam(params=discriminator.trainable_params(), learning_rate=lr)
truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
truncated_vgg19.set_train(False)

# Loss functions
content_loss_criterion = nn.MSELoss()
adversarial_loss_criterion = nn.BCEWithLogitsLoss()

def forward_fn_g(low_img, high_img):
    """forward fn of generator"""
    super_img = generator(low_img)
    super_img = convert_image(super_img, source='[-1, 1]', target='imagenet-norm')

    super_img_in_vgg_space = truncated_vgg19(super_img)
    high_img_in_vgg_space = truncated_vgg19(high_img)

    super_discriminated = discriminator(super_img)
    content_loss = content_loss_criterion(super_img_in_vgg_space, high_img_in_vgg_space)
    adversarial_loss = adversarial_loss_criterion(super_discriminated, ops.ones_like(super_discriminated))
    perceptual_loss = content_loss + beta * adversarial_loss

    return perceptual_loss, content_loss, adversarial_loss

def forward_fn_d(low_img, high_img):
    """forward fn for discriminator"""
    super_img = generator(low_img)
    high_discriminated = discriminator(high_img)
    super_discriminated = discriminator(super_img)

    adversarial_loss = adversarial_loss_criterion(super_discriminated, ops.zeros_like(super_discriminated)) + \
                       adversarial_loss_criterion(high_discriminated, ops.ones_like(high_discriminated))

    return adversarial_loss

grad_fn_g = mindspore.value_and_grad(forward_fn_g, None, generator.trainable_params(), True)
grad_fn_d = mindspore.value_and_grad(forward_fn_d, None, discriminator.trainable_params())

def train_step(low_img, high_img):
    """train step"""
    (_, content_loss, adversarial_loss_g), grad_g = grad_fn_g(low_img, high_img)
    adversarial_loss_d, grad_d = grad_fn_d(low_img, high_img)

    if grad_clip is not None:
        grad_g = clip_gradient(grad_g, grad_clip)
        grad_d = clip_gradient(grad_d, grad_clip)

    optimizer_g(grad_g)
    optimizer_d(grad_d)

    return content_loss, adversarial_loss_g, adversarial_loss_d

def train(epoch):
    """train"""
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_c = AverageMeter()  # content loss
    losses_a = AverageMeter()  # adversarial loss in the generator
    losses_d = AverageMeter()  # adversarial loss in the discriminator

    generator.set_train()
    discriminator.set_train()

    start = time.time()
    for i, (lr_imgs, hr_imgs) in enumerate(train_dataset.create_tuple_iterator()):
        data_time.update(time.time() - start)

        content_loss, adversarial_loss_g, adversarial_loss_d = train_step(lr_imgs, hr_imgs)

        losses_c.update(float(content_loss), lr_imgs.shape[0])
        losses_a.update(float(adversarial_loss_g), lr_imgs.shape[0])
        losses_d.update(float(adversarial_loss_d), hr_imgs.shape[0])

        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_dataset)}]----'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  f'Cont. Loss {losses_c.val:.4f} ({losses_c.avg:.4f})----'
                  f'Adv. Loss {losses_a.val:.4f} ({losses_a.avg:.4f})----'
                  f'Disc. Loss {losses_d.val:.4f} ({losses_d.avg:.4f})')

def main():
    """main"""
    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
    save_checkpoint(generator, f'generator_epoch_{epoch}.ckpt')
    save_checkpoint(discriminator, f'discriminator_epoch_{epoch}.ckpt')

if __name__ == "__main__":
    main()

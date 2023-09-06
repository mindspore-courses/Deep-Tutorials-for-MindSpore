# pylint: disable=C0103
"""evaluate the model"""

from mindspore import load_checkpoint, load_param_into_net
from mindspore.dataset import GeneratorDataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model import SRResNet, Generator
from datasets import SRDataset
from utils import AverageMeter, convert_image

# Data
data_folder = "./"
test_data_names = ["Set5", "Set14", "BSDS100"]
scaling_factor = 4

# Model parameters
large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks = 16  # number of residual blocks

# Model checkpoints
srresnet_checkpoint = "./srresnet_epoch_0.ckpt"
generator_checkpoint = './generator_epoch_0.ckpt'

srresnet_params_dict = load_checkpoint(srresnet_checkpoint)
generator_params_dict = load_checkpoint(generator_checkpoint)


# Load model, either the SRResNet or the SRGAN(Generator)
srresnet = SRResNet(large_kernel_size=large_kernel_size,
                    small_kernel_size=small_kernel_size,
                    n_channels=n_channels, n_blocks=n_blocks,
                    scaling_factor=scaling_factor)
load_param_into_net(srresnet, srresnet_params_dict)
# model = srresnet
generator = Generator(large_kernel_size=large_kernel_size,
                      small_kernel_size=small_kernel_size,
                      n_channels=n_channels,
                      n_blocks=n_blocks,
                      scaling_factor=scaling_factor)

load_param_into_net(generator, generator_params_dict)
model = generator
model.set_train(False)

for test_data_name in test_data_names:
    print(f"\nFor {test_data_name}:\n")

    test_dataset_source = SRDataset(data_folder,
                                    split='test',
                                    crop_size=0,
                                    scaling_factor=4,
                                    lr_img_type='imagenet-norm',
                                    hr_img_type='[-1, 1]',
                                    test_data_name=test_data_name)
    test_dataset = GeneratorDataset(test_dataset_source, column_names=['lr_imgs', 'hr_imgs'])
    test_dataset = test_dataset.batch(1, num_parallel_workers=4)

    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

    for i, (lr_imgs, hr_imgs) in enumerate(test_dataset.create_tuple_iterator()):
        sr_imgs = model(lr_imgs)

        # Calculate PSNR and SSIM
        sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        psnr = peak_signal_noise_ratio(hr_imgs_y.asnumpy(), sr_imgs_y.asnumpy(), data_range=255.)
        ssim = structural_similarity(hr_imgs_y.asnumpy(), sr_imgs_y.asnumpy(), data_range=255.)
        PSNRs.update(psnr, lr_imgs.shape[0])
        SSIMs.update(ssim, lr_imgs.shape[0])

    # Print average PSNR and SSIM
    print(f'PSNR - {PSNRs.avg:.3f}')
    print(f'SSIM - {SSIMs.avg:.3f}')

print("\n")

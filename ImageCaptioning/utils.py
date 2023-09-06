# pylint: disable=C0103
# pylint: disable=E0401

"""utils"""
import os
import json
from collections import Counter
from random import seed, choice, sample
import mindspore
import h5py
import numpy as np
from mindspore import ops, save_checkpoint
from tqdm import tqdm
from imageio import imread
from PIL import Image

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    with open(karpathy_json_path, 'r', encoding='utf-8') as j:
        data = json.load(j)

    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

            if len(captions) == 0:
                continue

            path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
                image_folder, img['filename'])

            if img['split'] in {'train', 'restval'}:
                train_image_paths.append(path)
                train_image_captions.append(captions)
            elif img['split'] in {'val'}:
                val_image_paths.append(path)
                val_image_captions.append(captions)
            elif img['split'] in {'test'}:
                test_image_paths.append(path)
                test_image_captions.append(captions)

    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    words = [w[0] for w in word_freq.items() if w[1] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w', encoding='utf-8') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print(f"\nReading {split} images and captions, storing to file...\n")

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = np.array(Image.fromarray(img).resize((256, 256)))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w', encoding='utf-8') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w', encoding='utf-8') as j:
                json.dump(caplens, j)

def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.shape[1])
    return ops.uniform(embeddings, -bias, bias)

def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r', encoding='utf-8') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())
    embeddings = mindspore.Tensor((len(vocab), emb_dim), dtype=mindspore.float32)
    embeddings = init_embedding(embeddings)

    print("\nLoading embeddings...")
    for line in open(emb_file, 'r', encoding='utf-8'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(float, filter(lambda n: n and not n.isspace(), line[1:0])))

        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = mindspore.Tensor(embedding, dtype=mindspore.float32)

    return embeddings, emb_dim

def clip_gradient(grads, grad_clip):
    """clip gradient"""
    grads = ops.clip_by_value(grads, -grad_clip, grad_clip)
    return grads

class AverageMeter:
    '''
    AverageMeter
    '''
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        '''reset'''
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        '''update'''
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    new_lr = optimizer.learning_rate.value() * shrink_factor
    optimizer.learning_rate.set_data(new_lr)
    print("The new learning rate is {new_lr}")

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
    return correct_total * (100.0 / batch_size)

def save_model(data_name, epoch, decoder, is_best=False):
    """
    save model
    """

    # encoder_filename = f'encoder_{data_name}_{epoch}.ckpt'
    decoder_filename = f'decoder_{data_name}_{epoch}.ckpt'
    # save_checkpoint(encoder, encoder_filename)
    save_checkpoint(decoder, decoder_filename)
    if is_best:
        # save_checkpoint(encoder, 'BEST_' + encoder_filename)
        save_checkpoint(decoder, 'BEST_' + decoder_filename)

def adjust_shape(input_tensor, lengths):
    """
    adjust shape for criterion
    """

    packed_tensor = input_tensor[:, :max(lengths)]
    packed_tensor = packed_tensor.flatten(start_dim=0, end_dim=1)
    return packed_tensor

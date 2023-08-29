# pylint: disable=C0103
# pylint: disable=E0401

"""inference"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import mindspore
from matplotlib import cm
from imageio import imread
from PIL import Image
from mindspore import ops
from mindspore.dataset import vision
from mindspore import load_checkpoint, load_param_into_net
from models import Encoder, DecoderWithAttention

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    caption image beam search
    """
    k = beam_size
    vocab_size = len(word_map)

    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = np.array(Image.fromarray(img).resize((256, 256)))
    img = img.transpose(2, 0, 1)
    img = img / 255
    img = mindspore.Tensor(img, dtype=mindspore.int32)
    normalize = vision.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],
                                 is_hwc=False)
    image = normalize(img)

    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.shape[1]
    encoder_dim = encoder_out.hape[3]
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.shape[1]
    encoder_out = encoder_out.broadcast_to((k, num_pixels, encoder_dim))

    k_prev_words = mindspore.Tensor([[word_map['<start>']]] * k, dtype=mindspore.int32)
    seqs = k_prev_words
    top_k_scores = ops.zeros((k, 1))
    seqs_alpha = ops.ones((k, 1, enc_image_size, enc_image_size))

    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe
        h, c = decoder.decode_step(ops.cat([embeddings, awe], axis=1), (h, c))

        scores = decoder.fc(h)
        scores = ops.log_softmax(scores, axis=1)
        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size

        seqs = ops.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], axis=1)  # (s, step+1)
        seqs_alpha = ops.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               axis=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].asnumpy().tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].asnumpy().tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)

        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas

def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """visualize attention"""

    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]
    for t, word in enumerate(words):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words).astype(np.int32) / 5), 5, t + 1)

        plt.text(0, 1, f'{word}', color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()

def main():
    """main"""
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--checkpoint', '-ch', help='path to checkpoint')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()
    params_dict = load_checkpoint(args.checkpoint)

    with open(args.word_map, 'r', encoding='utf-8') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    encoder = Encoder()
    decoder = DecoderWithAttention(attention_dim=512,
                                   embed_dim=512,
                                   decoder_dim=512,
                                   vocab_size=len(word_map),
                                   dropout=0.5)
    load_param_into_net(decoder, params_dict)
    encoder.set_train(False)
    decoder.set_train(False)

    with open(args.word_map, 'r', encoding='utf-8') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = mindspore.Tensor(alphas, dtype=mindspore.float32)
    visualize_att(args.img, seq, alphas, rev_word_map, args.smooth)

if __name__ == "__main__":
    main()

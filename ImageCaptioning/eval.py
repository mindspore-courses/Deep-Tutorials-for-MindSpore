# pylint: disable=C0103
# pylint: disable=E0401

"""evaluation"""
import json
import mindspore
from tqdm import tqdm
from mindspore import ops
from mindspore.dataset import vision
from mindspore import load_checkpoint, load_param_into_net
from mindspore.dataset import GeneratorDataset
from models import Encoder, DecoderWithAttention
from nltk.translate.bleu_score import corpus_bleu
from datasets import CaptionDataset

data_folder = 'Deep-Tutorials-for-MindSpore/dataset_coco'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
checkpoint = 'decoder_coco_5_cap_per_img_5_min_word_freq_1.ckpt'
word_map_file = 'Deep-Tutorials-for-MindSpore/dataset_coco/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'

emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5

with open(word_map_file, 'r', encoding='utf-8') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

encoder = Encoder()
decoder = DecoderWithAttention(attention_dim=attention_dim,
                               embed_dim=emb_dim,
                               decoder_dim=decoder_dim,
                               vocab_size=vocab_size,
                               dropout=dropout)
params_dict = load_checkpoint(checkpoint)
load_param_into_net(decoder, params_dict)
encoder.set_train(False)
decoder.set_train(False)

normalize = vision.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225],
                             is_hwc=False)

def evaluate(beam_size):
    """evaluate"""
    test_dataset = GeneratorDataset(CaptionDataset(data_folder, data_name, 'TEST'), ['img', 'caption', 'caplen', 'allcaps'])
    test_dataset = test_dataset.map(operations=[normalize], input_columns='img')
    test_dataset = test_dataset.batch(1)

    references = []
    hypotheses = []

    with tqdm(total = len(test_dataset)) as progress:
        progress.set_description("EVALUATING AT BEAM SIZE " + str(beam_size))
        for i, (image, _, _, allcaps) in enumerate(test_dataset.create_tuple_iterator()):

            k = beam_size
            encoder_out = encoder(image)
            encoder_dim = encoder_out.shape[3]
            encoder_out = encoder_out.view(1, -1, encoder_dim)
            num_pixels = encoder_out.shape[1]
            encoder_out = encoder_out.broadcast_to((k, num_pixels, encoder_dim))

            k_prev_words = mindspore.Tensor([[word_map['<start>']]] * k, dtype=mindspore.int32)
            seqs = k_prev_words
            top_k_scores = ops.zeros((k, 1))

            complete_seqs = []
            complete_seqs_scores = []

            step = 1
            h, c = decoder.init_hidden_state(encoder_out)

            while True:

                embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

                awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

                gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
                awe = gate * awe

                h, c = decoder.decode_step(ops.cat([embeddings, awe], axis=1), (h, c))  # (s, decoder_dim)

                scores = decoder.fc(h)  # (s, vocab_size)
                scores = ops.log_softmax(scores, axis=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words / vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = ops.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], axis=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].asnumpy().tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
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

            # Referencess
            img_caps = allcaps[0].asnumpy().tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)

            # Hypotheses
            hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

            assert len(references) == len(hypotheses)
            progress.update(1)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    return bleu4

def main():
    """main"""
    beam_size = 3
    bleu4 = evaluate(beam_size)
    print(f"\nBLEU-4 score @ beam size of {beam_size} is {bleu4:.f}.")

if __name__ == '__main__':
    main()

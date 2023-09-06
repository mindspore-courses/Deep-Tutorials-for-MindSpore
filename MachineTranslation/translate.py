# pylint: disable=C0103
"""translate"""

import math
import youtokentome
import mindspore
from mindspore import ops, load_checkpoint, load_param_into_net
from model import Transformer
from utils import get_positional_encoding

bpe_model = youtokentome.BPE(model="/transformer data/bpe.model")

model = Transformer(vocab_size=37000,
                    positional_encoding=get_positional_encoding(512, 160),
                    d_model=512,
                    n_heads=8,
                    d_queries=64,
                    d_values=64,
                    d_inner=2048,
                    n_layers=6,
                    dropout=0.1)
params_dict = load_checkpoint('avaraged_transformer.ckpt')
load_param_into_net(model, params_dict)

def translate(source_sequence, beam_size=4, length_norm_coefficient=0.6):
    """
    Translates a source language sequence to the target language, with beam search decoding.
    """

    k = beam_size

    # Minimum number of hypotheses to complete
    n_completed_hypotheses = min(k, 10)

    # Vocab size
    vocab_size = bpe_model.vocab_size()

    # If the source sequence is a string, convert to a tensor of IDs
    if isinstance(source_sequence, str):
        encoder_sequences = bpe_model.encode(source_sequence,
                                             output_type=youtokentome.OutputType.ID,
                                             bos=False,
                                             eos=False)
        encoder_sequences = mindspore.Tensor(encoder_sequences, mindspore.int64).unsqueeze(0)  # (1, source_sequence_length)
    else:
        encoder_sequences = source_sequence
    encoder_sequence_lengths = mindspore.Tensor([encoder_sequences.shape[1]], mindspore.int64)

    # Encode
    encoder_sequences = model.encoder(encoder_sequences=encoder_sequences,
                                        encoder_sequence_lengths=encoder_sequence_lengths)  # (1, source_sequence_length, d_model)

    # Our hypothesis to begin with is just <BOS>
    hypotheses = mindspore.Tensor([[bpe_model.subword_to_id('<BOS>')]], mindspore.int64)  # (1, 1)
    hypotheses_lengths = mindspore.Tensor([hypotheses.shape[1]], mindspore.int64)  # (1)

    # Tensor to store hypotheses' scores; now it's just 0
    hypotheses_scores = ops.zeros(1)  # (1)

    # Lists to store completed hypotheses and their scores
    completed_hypotheses = []
    completed_hypotheses_scores = []

    # Start decoding
    step = 1

    while True:
        s = hypotheses.shape[0]
        decoder_sequences = model.decoder(decoder_sequences=hypotheses,
                                         decoder_sequence_lengths=hypotheses_lengths,
                                         encoder_sequences=encoder_sequences.repeat(s, 1, 1),
                                         encoder_sequence_lengths=encoder_sequence_lengths.repeat(s))
                                        # (s, step, vocab_size)

        # Scores at this step
        scores = decoder_sequences[:, -1, :]  # (s, vocab_size)
        scores = ops.log_softmax(scores, axis=-1)  # (s, vocab_size)

        # Add hypotheses' scores from last step to scores at this step to get scores for all possible new hypotheses
        scores = hypotheses_scores.unsqueeze(1) + scores  # (s, vocab_size)

        # Unroll and find top k scores, and their unrolled indices
        top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True)  # (k)

        # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
        prev_word_indices = unrolled_indices // vocab_size  # (k)
        next_word_indices = unrolled_indices % vocab_size  # (k)

        # Construct the the new top k hypotheses from these indices
        top_k_hypotheses = ops.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)],
                                        axis=1)  # (k, step + 1)

        # Which of these new hypotheses are complete (reached <EOS>)?
        complete = next_word_indices == bpe_model.subword_to_id('<EOS>')  # (k), bool

        # Set aside completed hypotheses and their scores normalized by their lengths
        # For the length normalization formula, see
        # "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
        completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
        norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
        completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

        # Stop if we have completed enough hypotheses
        if len(completed_hypotheses) >= n_completed_hypotheses:
            break

        # Else, continue with incomplete hypotheses
        hypotheses = top_k_hypotheses[~complete]  # (s, step + 1)
        hypotheses_scores = top_k_hypotheses_scores[~complete]  # (s)
        hypotheses_lengths = mindspore.Tensor(hypotheses.shape[0] * [hypotheses.size(1)], mindspore.int64)  # (s)

        # Stop if things have been going on for too long
        if step > 100:
            break
        step += 1

    # If there is not a single completed hypothesis, use partial hypotheses
    if len(completed_hypotheses) == 0:
        completed_hypotheses = hypotheses.asnumpy().tolist()
        completed_hypotheses_scores = hypotheses_scores.asnumpy().tolist()

    # Decode the hypotheses
    all_hypotheses = []
    for i, h in enumerate(bpe_model.decode(completed_hypotheses, ignore_ids=[0, 2, 3])):
        all_hypotheses.append({"hypothesis": h, "score": completed_hypotheses_scores[i]})

    # Find the best scoring completed hypothesis
    i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
    best_hypothesis = all_hypotheses[i]["hypothesis"]

    return best_hypothesis, all_hypotheses

if __name__ == '__main__':
    translate("Anyone who retains the ability to recognise beauty will never become old.")

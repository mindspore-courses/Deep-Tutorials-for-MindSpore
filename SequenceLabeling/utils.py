# pylint: disable=C0103
# pylint: disable=E0401
"""utils"""

import codecs
from collections import Counter
from functools import reduce
import numpy as np
import mindspore
from mindspore import ops


def read_words_tags(file, tag_ind, caseless=True):
    """
    Reads raw data in the CoNLL 2003 format and returns word and tag sequences.

    :param file: file with raw data in the CoNLL 2003 format
    :param tag_ind: column index of tag
    :param caseless: lowercase words?
    :return: word, tag sequences
    """
    with codecs.open(file, 'r', 'utf-8') as f:
        lines = f.readlines()
    words = []
    tags = []
    temp_w = []
    temp_t = []
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            feats = line.rstrip('\n').split()
            temp_w.append(feats[0].lower() if caseless else feats[0])
            temp_t.append(feats[tag_ind])
        elif len(temp_w) > 0:
            assert len(temp_w) == len(temp_t)
            words.append(temp_w)
            tags.append(temp_t)
            temp_w = []
            temp_t = []
    # last sentence
    if len(temp_w) > 0:
        assert len(temp_w) == len(temp_t)
        words.append(temp_w)
        tags.append(temp_t)

    # Sanity check
    assert len(words) == len(tags)

    return words, tags


def create_maps(words, tags, min_word_freq=5, min_char_freq=1):
    """
    Creates word, char, tag maps.

    :param words: word sequences
    :param tags: tag sequences
    :param min_word_freq: words that occur fewer times than this threshold are binned as <unk>s
    :param min_char_freq: characters that occur fewer times than this threshold are binned as <unk>s
    :return: word, char, tag maps
    """
    word_freq = Counter()
    char_freq = Counter()
    tag_map = set()
    for w, t in zip(words, tags):
        word_freq.update(w)
        char_freq.update(list(reduce(lambda x, y: list(x) + [' '] + list(y), w)))
        tag_map.update(t)

    word_map = {k: v + 1 for v, k in enumerate([w for w, freq in word_freq.items() if freq > min_word_freq])}
    char_map = {k: v + 1 for v, k in enumerate([c for c, freq in char_freq.items() if freq > min_char_freq])}
    tag_map = {k: v + 1 for v, k in enumerate(tag_map)}

    word_map['<pad>'] = 0
    word_map['<end>'] = len(word_map)
    word_map['<unk>'] = len(word_map)
    char_map['<pad>'] = 0
    char_map['<end>'] = len(char_map)
    char_map['<unk>'] = len(char_map)
    tag_map['<pad>'] = 0
    tag_map['<start>'] = len(tag_map)
    tag_map['<end>'] = len(tag_map)

    return word_map, char_map, tag_map


def create_input_array(words, tags, word_map, char_map, tag_map):
    """
    Creates input numpy array that will be used to create a Dataset.

    :param words: word sequences
    :param tags: tag sequences
    :param word_map: word map
    :param char_map: character map
    :param tag_map: tag map
    :return: padded encoded words, padded encoded forward chars, padded encoded backward chars,
            padded forward character markers, padded backward character markers, padded encoded tags,
            word sequence lengths, char sequence lengths
    """
    # Encode sentences into word maps with <end> at the end
    wmaps = list(map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [word_map['<end>']], words))

    # Forward and backward character streams
    chars_f = list(map(lambda s: list(reduce(lambda x, y: list(x) + [' '] + list(y), s)) + [' '], words))
    chars_b = list(
        map(lambda s: list(reversed([' '] + list(reduce(lambda x, y: list(x) + [' '] + list(y), s)))), words))

    # Encode streams into forward and backward character maps with <end> at the end
    cmaps_f = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_f))
    cmaps_b = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_b))

    # Positions of spaces and <end> character
    # Words are predicted or encoded at these places in the language and tagging models respectively
    cmarkers_f = list(map(lambda s: [ind for ind in range(len(s)) if s[ind] == char_map[' ']] + [len(s) - 1], cmaps_f))
    # Reverse the markers for the backward stream before adding <end>, so the words of the f and b markers coincide
    cmarkers_b = list(
        map(lambda s: list(reversed([ind for ind in range(len(s)) if s[ind] == char_map[' ']])) + [len(s) - 1],
            cmaps_b))

    # Encode tags into tag maps with <end> at the end
    tmaps = list(map(lambda s: list(map(lambda t: tag_map[t], s)) + [tag_map['<end>']], tags))
    tmaps = list(map(lambda s: [tag_map['<start>'] * len(tag_map) + s[0]] + [s[i - 1] * len(tag_map) + s[i] for i in
                                                                             range(1, len(s))], tmaps))
    # Note - the actual tag indices can be recovered with tmaps % len(tag_map)

    # Pad, because need fixed length to be passed around by DataLoaders and other layers
    word_pad_len = max(list(map(len, wmaps)))
    char_pad_len = max(list(map(len, cmaps_f)))

    # Sanity check
    assert word_pad_len == max(list(map(len, tmaps)))

    padded_wmaps = []
    padded_cmaps_f = []
    padded_cmaps_b = []
    padded_cmarkers_f = []
    padded_cmarkers_b = []
    padded_tmaps = []
    wmap_lengths = []
    cmap_lengths = []

    for w, cf, cb, cmf, cmb, t in zip(wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps):
        # Sanity  checks
        assert len(w) == len(cmf) == len(cmb) == len(t)
        assert len(cmaps_f) == len(cmaps_b)

        # Pad
        # A note -  it doesn't really matter what we pad with, as long as it's a valid index
        # i.e., we'll extract output at those pad points (to extract equal lengths), but never use them

        padded_wmaps.append(w + [word_map['<pad>']] * (word_pad_len - len(w)))
        padded_cmaps_f.append(cf + [char_map['<pad>']] * (char_pad_len - len(cf)))
        padded_cmaps_b.append(cb + [char_map['<pad>']] * (char_pad_len - len(cb)))

        # 0 is always a valid index to pad markers with (-1 is too but torch.gather has some issues with it)
        padded_cmarkers_f.append(cmf + [0] * (word_pad_len - len(w)))
        padded_cmarkers_b.append(cmb + [0] * (word_pad_len - len(w)))

        padded_tmaps.append(t + [tag_map['<pad>']] * (word_pad_len - len(t)))

        wmap_lengths.append(len(w))
        cmap_lengths.append(len(cf))

        # Sanity check
        assert len(padded_wmaps[-1]) == len(padded_tmaps[-1]) == len(padded_cmarkers_f[-1]) == len(
            padded_cmarkers_b[-1]) == word_pad_len
        assert len(padded_cmaps_f[-1]) == len(padded_cmaps_b[-1]) == char_pad_len

    padded_wmaps = np.array(padded_wmaps, dtype=np.int64)
    padded_cmaps_f = np.array(padded_cmaps_f, dtype=np.int64)
    padded_cmaps_b = np.array(padded_cmaps_b, dtype=np.int64)
    padded_cmarkers_f = np.array(padded_cmarkers_f, dtype=np.int64)
    padded_cmarkers_b = np.array(padded_cmarkers_b, dtype=np.int64)
    padded_tmaps = np.array(padded_tmaps, dtype=np.int64)
    wmap_lengths = np.array(wmap_lengths, dtype=np.int64)
    cmap_lengths = np.array(cmap_lengths, dtype=np.int64)

    return padded_wmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, \
           wmap_lengths, cmap_lengths


def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.

    :param input_embedding: embedding tensor
    :return:
    """
    bias = mindspore.Tensor(np.sqrt(3.0 / input_embedding.shape[1]), dtype=input_embedding.dtype)
    input_embedding = ops.uniform(input_embedding.shape, -bias, bias, dtype=input_embedding.dtype)
    return input_embedding


def load_embeddings(emb_file, word_map, expand_vocab=True):
    """
    Load pre-trained embeddings for words in the word map.

    :param emb_file: file with pre-trained embeddings (in the GloVe format)
    :param word_map: word map
    :param expand_vocab: expand vocabulary of word map to vocabulary of pre-trained embeddings?
    :return: embeddings for words in word map, (possibly expanded) word map,
            number of words in word map that are in-corpus (subject to word frequency threshold)
    """
    with open(emb_file, 'r', encoding='utf-8') as f:
        emb_len = len(f.readline().split(' ')) - 1

    print(f"Embedding length is {emb_len}.")

    # Create tensor to hold embeddings for words that are in-corpus
    ic_embs = ops.randn(len(word_map), emb_len, dtype=mindspore.float32)
    init_embedding(ic_embs)

    if expand_vocab:
        print("You have elected to include embeddings that are out-of-corpus.")
        ooc_words = []
        ooc_embs = []
    else:
        print("You have elected NOT to include embeddings that are out-of-corpus.")

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r', encoding='utf-8'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(float, filter(lambda n: n and not n.isspace(), line[1:])))

        if not expand_vocab and emb_word not in word_map:
            continue

        # If word is in train_vocab, store at the correct index (as in the word_map)
        if emb_word in word_map:
            ic_embs[word_map[emb_word]] = mindspore.Tensor(embedding, dtype=mindspore.float32)

        # If word is in dev or test vocab, store it and its embedding into lists
        elif expand_vocab:
            ooc_words.append(emb_word)
            ooc_embs.append(embedding)

    lm_vocab_size = len(word_map)  # keep track of lang. model's output vocab size (no out-of-corpus words)

    if expand_vocab:
        print("'word_map' is being updated accordingly.")
        for word in ooc_words:
            word_map[word] = len(word_map)
        ooc_embs = mindspore.Tensor(np.array(ooc_embs), dtype=mindspore.float32)
        embeddings = ops.cat([ic_embs, ooc_embs], 0)

    else:
        embeddings = ic_embs

    # Sanity check
    assert embeddings.shape[0] == len(word_map)

    print(f"\nDone.\n Embedding vocabulary: {len(word_map)}\n Language Model vocabulary: {lm_vocab_size}.\n")

    return embeddings, word_map, lm_vocab_size


def clip_gradient(grads, grad_clip):
    """clip gradient"""
    grads = ops.clip_by_value(grads, -grad_clip, grad_clip)
    return grads


class AverageMeter:
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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


def adjust_learning_rate(optimizer, new_lr):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param new_lr: new learning rate
    """

    print("\nDECAYING learning rate.")
    ops.assign(optimizer.learning_rate, mindspore.Parameter(new_lr, mindspore.float32))
    print(f"The new learning rate is {optimizer.learning_rate}\n" )


def log_sum_exp(tensor, axis):
    """
    Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.

    :param tensor: tensor
    :param dim: dimension to calculate log-sum-exp of
    :return: log-sum-exp
    """
    m, _ = ops.max(tensor, axis)
    m_expanded = m.unsqueeze(axis).expand_as(tensor)
    return m + ops.log(ops.sum(ops.exp(tensor - m_expanded), axis))

def adjust_shape(input_tensor, lengths):
    """
    pack the padding of sequence
    """

    output_tensor = input_tensor[:, :max(lengths)]
    output_tensor = output_tensor.flatten(start_dim=0, end_dim=1)
    return output_tensor

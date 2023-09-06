# pylint: disable=C0103
"""utils"""

import os
import pickle
import json
import logging
import itertools
from collections import Counter
import gensim
import pandas as pd
import numpy as np
import mindspore
from tqdm import tqdm
from mindspore import ops
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer

classes = ['Society & Culture',
           'Science & Mathematics',
           'Health',
           'Education & Reference',
           'Computers & Internet',
           'Sports',
           'Business & Finance',
           'Entertainment & Music',
           'Family & Relationships',
           'Politics & Government']

label_map = {k: v for v, k in enumerate(classes)}
rev_label_map = {v: k for k, v in label_map.items()}

# Tokenizers
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()

def preprocess(text):
    """
    Pre-process text for use in the model. This includes lower-casing, standardizing newlines, removing junk.
    """
    if isinstance(text, float):
        return ''

    return text.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace('&#xd;', '\n')

def read_csv(csv_folder, split, sentence_limit, word_limit):
    """
    Read CSVs containing raw training data, clean documents and labels, and do a word-count.

    :param csv_folder: folder containing the CSV
    :param split: train or test CSV?
    :param sentence_limit: truncate long documents to these many sentences
    :param word_limit: truncate long sentences to these many words
    :return: documents, labels, a word-count
    """
    assert split in {'train', 'test'}

    docs = []
    labels = []
    word_counter = Counter()
    data = pd.read_csv(os.path.join(csv_folder, split + '.csv'), header=None)
    for i in tqdm(range(data.shape[0])):
        row = list(data.loc[i, :])

        sentences = []

        for text in row[1:]:
            for paragraph in preprocess(text).splitlines():
                sentences.extend(list(sent_tokenizer.tokenize(paragraph)))

        words = []
        for s in sentences[:sentence_limit]:
            w = word_tokenizer.tokenize(s)[:word_limit]
            # If sentence is empty (due to removing punctuation, digits, etc.)
            if len(w) == 0:
                continue
            words.append(w)
            word_counter.update(w)
        # If all sentences were empty
        if len(words) == 0:
            continue

        labels.append(int(row[0]) - 1)  # since labels are 1-indexed in the CSV
        docs.append(words)

    return docs, labels, word_counter

def create_input_files(csv_folder, output_folder, sentence_limit, word_limit, min_word_count=5,
                       save_word2vec_data=True):
    """
    Create data files to be used for training the model.

    :param csv_folder: folder where the CSVs with the raw data are located
    :param output_folder: folder where files must be created
    :param sentence_limit: truncate long documents to these many sentences
    :param word_limit: truncate long sentences to these many words
    :param min_word_count: discard rare words which occur fewer times than this number
    :param save_word2vec_data: whether to save the data required for training word2vec embeddings
    """
    # Read training data
    print('\nReading and preprocessing training data...\n')
    train_docs, train_labels, word_counter = read_csv(csv_folder, 'train', sentence_limit, word_limit)

    # Save text data for word2vec
    if save_word2vec_data:
        with open(os.path.join(output_folder, 'word2vec_data.pkl'), 'wb') as f:
            pickle.dump(train_docs, f)
        print(f'\nText data for word2vec saved to {os.path.abspath(output_folder)}.\n')

    # Create word map
    word_map = {}
    word_map['<pad>'] = 0
    for word, count in word_counter.items():
        if count >= min_word_count:
            word_map[word] = len(word_map)
    word_map['<unk>'] = len(word_map)
    print(f'\nDiscarding words with counts less than {min_word_count}, the size of the vocabulary is {len(word_map)}.\n')

    with open(os.path.join(output_folder, 'word_map.json'), 'w', encoding='utf-8') as j:
        json.dump(word_map, j)
    print(f'Word map saved to {os.path.abspath(output_folder)}.\n')

    # Encode and pad
    print('Encoding and padding training data...\n')
    encoded_train_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), train_docs))
    sentences_per_train_document = list(map(len, train_docs))
    words_per_train_sentence = list(
        map(lambda doc: list(map(len, doc)) + [0] * (sentence_limit - len(doc)), train_docs))

    # Save
    print('Saving...\n')
    assert len(encoded_train_docs) == len(train_labels) == len(sentences_per_train_document) == len(
        words_per_train_sentence)
    # Because of the large data, saving as a JSON can be very slow
    with open(os.path.join(output_folder, 'TRAIN_data.pkl'), 'wb') as f:
        pickle.dump({'docs': encoded_train_docs,
                     'labels': train_labels,
                     'sentences_per_document': sentences_per_train_document,
                     'words_per_sentence': words_per_train_sentence}, f)

    print(f'Encoded, padded training data saved to {os.path.abspath(output_folder)}.\n')

    # Free some memory
    del train_docs, encoded_train_docs, train_labels, sentences_per_train_document, words_per_train_sentence

    # Read test data
    print('Reading and preprocessing test data...\n')
    test_docs, test_labels, _ = read_csv(csv_folder, 'test', sentence_limit, word_limit)

    # Encode and pad
    print('\nEncoding and padding test data...\n')
    encoded_test_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), test_docs))
    sentences_per_test_document = list(map(len, test_docs))
    words_per_test_sentence = list(
        map(lambda doc: list(map(len, doc)) + [0] * (sentence_limit - len(doc)), test_docs))

    # Save
    print('Saving...\n')
    assert len(encoded_test_docs) == len(test_labels) == len(sentences_per_test_document) == len(
        words_per_test_sentence)

    with open(os.path.join(output_folder, 'TEST_data.pkl'), 'wb') as f:
        pickle.dump({'docs': encoded_test_docs,
                     'labels': test_labels,
                     'sentences_per_document': sentences_per_test_document,
                     'words_per_sentence': words_per_test_sentence}, f)

    print(f'Encoded, padded test data saved to {os.path.abspath(output_folder)}.\n')
    print('All done!\n')

def train_word2vec_model(data_folder, algorithm='skipgram'):
    """
    Train a word2vec model for word embeddings.

    See the paper by Mikolov et. al. for details - https://arxiv.org/pdf/1310.4546.pdf

    :param data_folder: folder with the word2vec training data
    :param algorithm: use the Skip-gram or Continous Bag Of Words (CBOW) algorithm?
    """
    assert algorithm in ['skipgram', 'cbow']
    sg = 1 if algorithm == 'skipgram' else 0

    # Read data
    with open(os.path.join(data_folder, 'word2vec_data.pkl'), 'rb') as f:
        sentences = pickle.load(f)

    sentences = list(itertools.chain.from_iterable(sentences))

    # Activate logging for verbose training
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Initialize and train the model (this will take some time)
    model = gensim.models.word2vec.Word2Vec(sentences=sentences, size=200, workers=8, window=10, min_count=5,
                                            sg=sg)

    # Normalize vectors and save model
    model.init_sims(True)
    model.wv.save(os.path.join(data_folder, 'word2vec_model'))

def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.shape[1])
    return ops.uniform(embeddings.shape, -bias, bias)

def load_word2vec_embeddings(word2vec_file, word_map):
    """
    Load pre-trained embeddings for words in the word map.

    :param word2vec_file: location of the trained word2vec model
    :param word_map: word map
    :return: embeddings for words in the word map, embedding size
    """
    # Load word2vec model into memory
    w2v = gensim.models.KeyedVectors.load(word2vec_file, mmap='r')

    print(f"\nEmbedding length is {w2v.vector_size}.\n")

    # Create tensor to hold embeddings for words that are in-corpus
    embeddings = ops.zeros((len(word_map), w2v.vector_size), dtype=mindspore.float32)
    embeddings = init_embedding(embeddings)

    # Read embedding file
    print("Loading embeddings...")
    for word in word_map:
        if word in w2v.vocab:
            embeddings[word_map[word]] = mindspore.Tensor(w2v[word], dtype=mindspore.float32)

    print(f"Done.\n Embedding vocabulary: {len(word_map)}.\n")

    return embeddings, w2v.vector_size


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

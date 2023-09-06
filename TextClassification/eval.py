# pylint: disable=C0103
"""evaluate HierarchialAttentionNetwork"""

import os
import json
from tqdm import tqdm
from mindspore.dataset import GeneratorDataset
from mindspore import ops, load_checkpoint, load_param_into_net
from utils import label_map, load_word2vec_embeddings, AverageMeter
from model import HierarchialAttentionNetwork
from datasets import HANDataset

data_folder = '/han data'

# Evaluation parameters
batch_size = 64  # batch size
workers = 4  # number of workers
print_freq = 2000  # print training or validation status every __ batches
checkpoint = 'checkpoint_han_epoch_0.ckpt'

word2vec_file = os.path.join(data_folder, 'word2vec_model')  # path to pre-trained word2vec embeddings
with open(os.path.join(data_folder, 'word_map.json'), 'r', encoding='utf-8') as j:
    word_map = json.load(j)

n_classes = len(label_map)
word_rnn_size = 50  # word RNN size
sentence_rnn_size = 50  # character RNN size
word_rnn_layers = 1  # number of layers in character RNN
sentence_rnn_layers = 1  # number of layers in word RNN
word_att_size = 100  # size of the word-level attention layer (also the size of the word context vector)
sentence_att_size = 100  # size of the sentence-level attention layer (also the size of the sentence context vector)
dropout = 0.3  # dropout

_, emb_size = load_word2vec_embeddings(word2vec_file, word_map)

model = HierarchialAttentionNetwork(n_classes=n_classes,
                                    vocab_size=len(word_map),
                                    emb_size=emb_size,
                                    word_rnn_size=word_rnn_size,
                                    sentence_rnn_size=sentence_rnn_size,
                                    word_rnn_layers=word_rnn_layers,
                                    sentence_rnn_layers=sentence_rnn_layers,
                                    word_att_size=word_att_size,
                                    sentence_att_size=sentence_att_size,
                                    dropout=dropout)

params_dict = load_checkpoint(checkpoint)
load_param_into_net(model, params_dict)
model.set_train(False)

test_dataset_source = HANDataset(data_folder, 'test')
test_dataset = GeneratorDataset(test_dataset_source,
                                ['docs', 'sentences_per_document', 'words_per_sentence', 'labels'])
test_dataset = test_dataset.batch(batch_size, num_parallel_workers=workers)

accs = AverageMeter()
for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(
        tqdm(test_dataset.create_tuple_iterator(), desc='Evaluating')):

    sentences_per_document = sentences_per_document.squeeze(1)
    scores, _, _ = model(documents, sentences_per_document, words_per_sentence)

    # Find accuracy
    _, predictions = scores.max(axis=1)  # (n_documents)
    correct_predictions = float(ops.equal(predictions, labels).sum())
    accuracy = correct_predictions / labels.shape[0]

    accs.update(accuracy, labels.shape[0])

print(f'\nTEST ACCURACY - {accs.avg * 100.:1f} per cent\n')

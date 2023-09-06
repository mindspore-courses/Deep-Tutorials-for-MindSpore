# pylint: disable=C0103
"""train model HierarchialAttentionNetwork"""

import os
import json
import time
import mindspore
from mindspore import nn, ops
from mindspore import save_checkpoint
from mindspore.dataset import GeneratorDataset
from model import HierarchialAttentionNetwork
from datasets import HANDataset
from utils import label_map, load_word2vec_embeddings, clip_gradient, AverageMeter

# Data parameters
data_folder = '/han data'
word2vec_file = os.path.join(data_folder, 'word2vec_model')  # path to pre-trained word2vec embeddings
with open(os.path.join(data_folder, 'word_map.json'), 'r', encoding='utf-8') as j:
    word_map = json.load(j)

# Model parameters
n_classes = len(label_map)
word_rnn_size = 50  # word RNN size
sentence_rnn_size = 50  # character RNN size
word_rnn_layers = 1  # number of layers in character RNN
sentence_rnn_layers = 1  # number of layers in word RNN
word_att_size = 100  # size of the word-level attention layer (also the size of the word context vector)
sentence_att_size = 100  # size of the sentence-level attention layer (also the size of the sentence context vector)
dropout = 0.3  # dropout
fine_tune_word_embeddings = True  # fine-tune word embeddings?

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 64  # batch size
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
workers = 4  # number of workers
epochs = 2  # number of epochs to run
grad_clip = None  # clip gradients at this value
print_freq = 2000  # print training or validation status every __ batches
checkpoint = None  # path to model checkpoint, None if none


embeddings, emb_size = load_word2vec_embeddings(word2vec_file, word_map)

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

model.sentence_attention.word_attention.init_embeddings(embeddings)
model.sentence_attention.word_attention.fine_tune_embeddings(fine_tune_word_embeddings)
optimizer = nn.Adam(params=model.trainable_params(), learning_rate=lr)

criterion = nn.CrossEntropyLoss()

train_dataset_source = HANDataset(data_folder, 'train')
train_dataset = GeneratorDataset(train_dataset_source,
                ['docs', 'sentences_per_document', 'words_per_sentence', 'labels'])
train_dataset = train_dataset.batch(batch_size, num_parallel_workers=workers)

def foward_fn(documents, sentences_per_document, words_per_sentence, labels):
    """forward fn"""
    scores, _, _ = model(documents, sentences_per_document, words_per_sentence)
    loss = criterion(scores, labels)
    return loss, scores

grad_fn = mindspore.value_and_grad(foward_fn, None, model.trainable_params(), has_aux=True)

def train_step(documents, sentences_per_document, words_per_sentence, labels):
    """train step"""
    (loss, scores), grads = grad_fn(documents, sentences_per_document, words_per_sentence, labels)

    if grad_clip is not None:
        grads = clip_gradient(grads, grad_clip)

    optimizer(grads)

    return loss, scores

def train(epoch):
    """train"""

    model.set_train()

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    start = time.time()

    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(train_dataset.create_tuple_iterator()):

        data_time.update(time.time() - start)
        sentences_per_document = sentences_per_document.squeeze(1)

        loss, scores = train_step(documents, sentences_per_document, words_per_sentence, labels)

        _, predictions = scores.max(axis=1)
        correct_predictions = float(ops.equal(predictions, labels).sum())
        accuray = correct_predictions / labels.shape[0]

        losses.update(float(loss.item(), labels.shape[0]))
        batch_time.update(time.time() - start)
        accs.update(accuray, labels.shape[0])

        start = time.time()

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_dataset)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Accuracy {accs.val:.3f} ({accs.avg:.3f})')

def main():
    """main"""
    for epoch in range(start_epoch, start_epoch+epochs):
        train(epoch)
        save_checkpoint(model, f'checkpoint_han_epoch_{epoch}.ckpt')

if __name__ == '__main__':
    main()

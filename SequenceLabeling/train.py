# pylint: disable=C0103
# pylint: disable=E0401
"""train the model"""

import time
import os
import mindspore
from mindspore import nn, ops
from mindspore import load_checkpoint, load_param_into_net
from mindspore.dataset import GeneratorDataset
from utils import AverageMeter, create_input_array, adjust_shape
from utils import read_words_tags, create_maps, load_embeddings, clip_gradient
from models import LM_LSTM_CRF, ViterbiLoss
from datasets import WCDataset
from inference import ViterbiDecoder
from sklearn.metrics import f1_score
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Data parameters
task = 'ner'  # tagging task, to choose column in CoNLL 2003 dataset
train_file = 'Deep-Tutorials-for-MindSpore/dataset_conll2003/train.txt'  # path to training data
val_file = 'Deep-Tutorials-for-MindSpore/dataset_conll2003/valid.txt'  # path to validation data
test_file = 'Deep-Tutorials-for-MindSpore/dataset_conll2003/test.txt'  # path to test data
emb_file = 'Deep-Tutorials-for-MindSpore/glove6B/glove.6B.100d.txt'  # path to pre-trained word embeddings
min_word_freq = 5  # threshold for word frequency
min_char_freq = 1  # threshold for character frequency
caseless = True  # lowercase everything?
expand_vocab = True # expand model's input vocabulary to the pre-trained embeddings' vocabulary?

# Model parameters
char_emb_dim = 30  # character embedding size
with open(emb_file, 'r', encoding='utf-8') as f:
    word_emb_dim = len(f.readline().split(' ')) - 1  # word embedding size
word_rnn_dim = 300  # word RNN size
char_rnn_dim = 300  # character RNN size
char_rnn_layers = 1  # number of layers in character RNN
word_rnn_layers = 1  # number of layers in word RNN
highway_layers = 1  # number of layers in highway network
dropout = 0.5  # dropout
fine_tune_word_embeddings = False  # fine-tune pre-trained word embeddings?

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 10  # batch size
lr = 0.015  # learning rate
lr_decay = 0.05  # decay learning rate by this amount
momentum = 0.9  # momentum
workers = 1  # number of workers for loading data in the DataLoader
epochs = 200  # number of epochs to run without early-stopping
grad_clip = 5.  # clip gradients at this value
print_freq = 100  # print training or validation status every __ batches
best_f1 = 0.  # F1 score to start with
checkpoint = None  # path to model checkpoint, None if none

tag_ind = 1 if task == 'pos' else 3  # choose column in CoNLL 2003 dataset
tag_ind = 1 if task == 'pos' else 3  # choose column in CoNLL 2003 dataset

train_words, train_tags = read_words_tags(train_file, tag_ind, caseless)
val_words, val_tags = read_words_tags(val_file, tag_ind, caseless)


word_map, char_map, tag_map = create_maps(train_words + val_words, train_tags + val_tags, min_word_freq,
                                            min_char_freq)  # create word, char, tag maps
embeddings, word_map, lm_vocab_size = load_embeddings(emb_file, word_map,
                                                          expand_vocab)  # load pre-trained embeddings
model = LM_LSTM_CRF(tagset_size=len(tag_map),
                    charset_size=len(char_map),
                    char_emb_dim=char_emb_dim,
                    char_rnn_dim=char_rnn_dim,
                    char_rnn_layers=char_rnn_layers,
                    vocab_size=len(word_map),
                    lm_vocab_size=lm_vocab_size,
                    word_emb_dim=word_emb_dim,
                    word_rnn_dim=word_rnn_dim,
                    word_rnn_layers=word_rnn_layers,
                    dropout=dropout,
                    highway_layers=highway_layers)
if checkpoint is not None:
    param_dict = load_checkpoint(checkpoint)
    load_param_into_net(model, param_dict)
else:
    model.init_word_embeddings(embeddings)
    model.fine_tune_word_embeddings(fine_tune_word_embeddings)

optimizer = nn.SGD(params=model.trainable_params(), learning_rate=lr, momentum=momentum)

lm_criterion = nn.CrossEntropyLoss(ignore_index=0)
crf_criterion = ViterbiLoss(tag_map)

temp_word_map = {k: v for k, v in word_map.items() if v <= word_map['<unk>']}
train_inputs = create_input_array(train_words, train_tags, temp_word_map, char_map, tag_map)
val_inputs = create_input_array(val_words, val_tags, temp_word_map, char_map, tag_map)
column_names = ['wmaps', 'cmaps_f', 'cmaps_b', 'cmarkers_f', 'cmarkers_b', 'tmaps', 'wmap_lengths', 'cmap_lengths']
train_dataset = GeneratorDataset(WCDataset(*train_inputs), column_names)
val_dataset = GeneratorDataset(WCDataset(*val_inputs), column_names)
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

vb_decoder = ViterbiDecoder(tag_map)
def foward_fn(cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps, tmaps, wmap_lengths, cmap_lengths):
    """forward fn"""
    crf_scores, lm_f_scores, lm_b_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, _, _ = model(cmaps_f,
                                                                                                        cmaps_b,
                                                                                                        cmarkers_f,
                                                                                                        cmarkers_b,
                                                                                                        wmaps,
                                                                                                        tmaps,
                                                                                                        wmap_lengths,
                                                                                                        cmap_lengths)

    lm_lengths = wmap_lengths_sorted - 1
    lm_lengths = lm_lengths.asnumpy().tolist()

    lm_f_scores =(lm_f_scores, lm_lengths)
    lm_b_scores = adjust_shape(lm_b_scores, lm_lengths)

    lm_f_targets = wmaps_sorted[:, 1:]
    lm_f_targets = adjust_shape(lm_f_targets, lm_lengths)

    lm_b_targets = ops.cat(
        [mindspore.Tensor([word_map['<end>']] * wmaps_sorted.shape[0], dtype=mindspore.int32).unsqueeze(1), wmaps_sorted], axis=1
    )
    lm_b_targets = adjust_shape(lm_b_targets, lm_lengths) * lm_f_targets.bool()
    ce_loss = lm_criterion(lm_f_scores, lm_f_targets) + lm_criterion(lm_b_scores, lm_b_targets)
    vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)
    loss = ce_loss + vb_loss

    return loss, ce_loss, vb_loss, tmaps_sorted, wmap_lengths_sorted, crf_scores, lm_lengths

grad_fn = mindspore.value_and_grad(foward_fn, None, model.trainable_params(), has_aux=True)

def train_step(cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps, tmaps, wmap_lengths, cmap_lengths):
    """train step"""
    (loss, ce_loss, vb_loss, tmaps_sorted, wmap_lengths_sorted, crf_scores, lm_lengths), grad = \
        grad_fn(cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps, tmaps, wmap_lengths, cmap_lengths)
    if grad_clip is not None:
        grad = clip_gradient(grad, grad_clip)

    optimizer(grad)
    return loss, ce_loss, vb_loss, tmaps_sorted, wmap_lengths_sorted, crf_scores, lm_lengths

def train(epoch):
    '''train model'''
    model.set_train()

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    ce_losses = AverageMeter()  # cross entropy loss
    vb_losses = AverageMeter()  # viterbi loss
    f1s = AverageMeter()  # f1 score

    start = time.time()
    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths) in enumerate(
        train_dataset.create_tuple_iterator()
    ):
        data_time.update(time.time() - start)

        max_word_len = max(wmap_lengths.asnumpy().tolist())
        max_char_len = max(cmap_lengths.asnumpy().tolist())

        wmaps = wmaps[:, :max_word_len]
        cmaps_f = cmaps_f[:, :max_char_len]
        cmaps_b = cmaps_b[:, :max_char_len]
        cmarkers_f = cmarkers_f[:, :max_word_len]
        cmarkers_b = cmarkers_b[:, :max_word_len]
        tmaps = tmaps[:, :max_word_len]
        _, ce_loss, vb_loss, tmaps_sorted, wmap_lengths_sorted, crf_scores, lm_lengths = \
            train_step(cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps, tmaps, wmap_lengths, cmap_lengths)
        decoded = vb_decoder.decode(crf_scores, wmap_lengths_sorted)
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size

        f1 = f1_score(tmaps_sorted.asnumpy().flatten(), decoded.asnumpy().flatten(), average='macro')
        ce_losses.update(float(ce_loss), sum(lm_lengths))
        vb_losses.update(float(vb_loss), crf_scores.shape[0])
        batch_time.update(time.time() - start)
        f1s.update(f1, sum(lm_lengths))

        start = time.time()
        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_dataset)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'CE Loss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
                  f'VB Loss {vb_losses.val:.4f} ({vb_losses.avg:.4f})\t'
                  f'F1 {f1s.val:.3f} ({f1s.avg:.3f})')

def main():
    """main"""
    for epoch in range(start_epoch, epochs):
        train(epoch)

if __name__ == '__main__':
    main()

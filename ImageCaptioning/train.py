# pylint: disable=C0103
# pylint: disable=E0401

"""train"""
import time
import os
import json
import mindspore
from mindspore import nn, ops
from mindspore import load_checkpoint, load_param_into_net
from mindspore.dataset import GeneratorDataset
from mindspore.dataset import vision
from utils import AverageMeter, adjust_shape, adjust_learning_rate, clip_gradient, save_model, accuracy
from models import Encoder, DecoderWithAttention
from datasets import CaptionDataset
from nltk.translate.bleu_score import corpus_bleu

data_folder = 'Deep-Tutorials-for-MindSpore/dataset_coco'
data_name = 'coco_5_cap_per_img_5_min_word_freq'

emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5

# Training parameters
start_epoch = 2
epochs = 2  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = {'decoder':'decoder_coco_5_cap_per_img_5_min_word_freq_1.ckpt'}  # path to checkpoint, None if none

word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r', encoding='utf-8') as j:
    word_map = json.load(j)

decoder = DecoderWithAttention(attention_dim=attention_dim,
                                embed_dim=emb_dim,
                                decoder_dim=decoder_dim,
                                vocab_size=len(word_map),
                                dropout=dropout)

encoder = Encoder()
encoder.fine_tune(fine_tune_encoder)

decoder = DecoderWithAttention(attention_dim=attention_dim,
                                embed_dim=emb_dim,
                                decoder_dim=decoder_dim,
                                vocab_size=len(word_map),
                                dropout=dropout)

params = load_checkpoint('/data1/had/decoder_coco_5_cap_per_img_5_min_word_freq_1.ckpt')
params_not_load = load_param_into_net(decoder, params)

encoder = Encoder()
encoder.fine_tune(fine_tune_encoder)
decoder_optimizer = nn.Adam(params=decoder.trainable_params(), learning_rate=decoder_lr)
encoder_optimizer = nn.Adam(params=encoder.trainable_params(), learning_rate=encoder_lr) if fine_tune_encoder else None

criterion = nn.CrossEntropyLoss(ignore_index=0)
normalize_op = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_hwc=False)
train_dataset = GeneratorDataset(CaptionDataset(data_folder, data_name, 'TRAIN'), ['img', 'caption', 'caplen'])
train_dataset = train_dataset.map(operations=[normalize_op], input_columns='img')
train_dataset = train_dataset.batch(batch_size)
val_dataset = GeneratorDataset(CaptionDataset(data_folder, data_name, 'VAL'), ['img', 'caption', 'caplen', 'all_captions'])
val_dataset = val_dataset.map(operations=[normalize_op], input_columns='img')
val_dataset = val_dataset.batch(batch_size)

def forward_fn(imgs, caps, caplens):
    """forward fn"""
    imgs = encoder(imgs)
    scores, caps_sorted, decode_lengths, alphas, _ = decoder(imgs, caps, caplens)
    targets = caps_sorted[:, 1:]
    scores = adjust_shape(scores, decode_lengths)
    targets = adjust_shape(targets, decode_lengths)
    loss = criterion(scores, targets.astype(mindspore.int32))
    loss += alpha_c * ((1. - alphas.sum(axis=1)) ** 2).mean()
    return loss, scores, targets, decode_lengths

grad_fn = mindspore.value_and_grad(forward_fn, None, decoder.trainable_params(), has_aux=True)

def train(epoch):
    """train"""
    decoder.set_train()
    encoder.set_train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter() # top5 accuracy

    start = time.time()

    #Batches
    for i, (imgs, caps, caplens) in enumerate(train_dataset.create_tuple_iterator()):
        data_time.update(time.time() - start)
        (loss, scores, targets, decode_lengths), grad = grad_fn(imgs, caps, caplens)

        # Clip gradients
        if grad_clip is not None:
            grad = clip_gradient(grad, grad_clip)

        decoder_optimizer(grad)
        if encoder_optimizer is not None:
            encoder_optimizer(grad)
        top5 = accuracy(scores, targets, 5)
        losses.update(float(loss), sum(decode_lengths))
        top5accs.update(float(top5), sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            with open('my_log.txt', 'a', encoding='utf-8') as f:
                f.write(f'Epoch:[{epoch}][{i}/{len(train_dataset)}]\t step Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\tTop-5 Acc {top5accs.val:.3f} ({top5accs.avg:.3f})\n')
                f.close()

def validate():
    """validate"""
    decoder.set_train(False)
    if encoder is not None:
        encoder.set_train(False)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = []  # references (true captions) for calculating BLEU-4 score
    hypotheses = []  # hypotheses (predictions)

    for i, (imgs, caps, caplens, allcaps) in enumerate(val_dataset.create_tuple_iterator()):
        if encoder is not None:
            imgs = encoder(imgs)

        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        scores_copy = scores.copy()

        loss = criterion(scores, targets)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq ==  0:
            print(f'Validation: [{i}/{len(val_dataset)}]\tBatch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {loss.val:.4f} ({loss.avg:.4f})\tTop-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\n')

        allcaps = allcaps[sort_ind]
        for k in range(allcaps.shape[0]):
            img_caps = allcaps[k].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['pad']}],
                    img_caps))
            references.append(img_captions)

        _, preds = ops.max(scores_copy, axis=2)
        preds = preds.tolist()
        temp_preds = []
        for i, pred in enumerate(preds):
            temp_preds.append(pred[:decode_lengths[i]])
        preds = temp_preds
        hypotheses.extend(preds)

        assert len(references) == len(hypotheses)

    bleu4 = corpus_bleu(references, hypotheses)

    print(f'\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu4}\n')

    return bleu4

if checkpoint is not None:
    decoder_checkpoint = load_checkpoint(checkpoint['decoder'])
    load_param_into_net(decoder, decoder_checkpoint)

for curr_epoch in range(start_epoch, start_epoch + epochs):
    if epochs_since_improvement == 20:
        break
    if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
        adjust_learning_rate(decoder_optimizer, 0.8)
        if fine_tune_encoder:
            adjust_learning_rate(encoder_optimizer, 0.8)

    train(curr_epoch)

    recent_bleu4 = validate()

    is_best = recent_bleu4 > best_bleu4
    best_bleu4 = max(recent_bleu4, best_bleu4)
    if not is_best:
        epochs_since_improvement += 1
        print(f"\nEpochs since last improvement: {epochs_since_improvement}\n")
    else:
        epochs_since_improvement = 0

    save_model(data_name, curr_epoch, decoder, is_best)

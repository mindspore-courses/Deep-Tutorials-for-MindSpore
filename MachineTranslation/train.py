#pylint: disable=C0103
"""train Transformer"""

import time
from tqdm import tqdm
import mindspore
from mindspore import nn
from mindspore import save_checkpoint
from model import Transformer, LabelSmoothedCE
from utils import get_positional_encoding, get_lr, Accumulator, AverageMeter
from data_generator import SequenceDataset

# Data parameters
data_folder = '/transformer data'  # folder with data files

# Model parameters
d_model = 512  # size of vectors throughout the transformer model
n_heads = 8  # number of heads in the multi-head attention
d_queries = 64  # size of query vectors (and also the size of the key vectors) in the multi-head attention
d_values = 64  # size of value vectors in the multi-head attention
d_inner = 2048  # an intermediate size in the position-wise FC
n_layers = 6  # number of layers in the Encoder and Decoder
dropout = 0.1  # dropout probability
positional_encoding = get_positional_encoding(d_model=d_model, max_length=160)
# positional encodings up to the maximum possible pad-length

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
tokens_in_batch = 2000  # batch size in target language tokens
accumulate_step = 25000 // tokens_in_batch  # perform a training step, i.e. update parameters, once every so many batches
print_frequency = 20  # print status once every so many steps
n_steps = 100000  # number of training steps
warmup_steps = 8000  # number of warmup steps where learning rate is increased linearly;
lr = get_lr(1, d_model, warmup_steps)
# see utils.py for learning rate schedule; twice the schedule in the paper, as in the official transformer repo.

start_epoch = 0  # start at this epoch
betas = (0.9, 0.98)  # beta coefficients in the Adam ptimizer
epsilon = 1e-9  # epsilon termo in the Adam optimizer
label_smoothing = 0.1  # label smoothing co-efficient in the Cross Entropy loss

train_dataset = SequenceDataset(data_folder=data_folder,
                                source_suffix="en",
                                target_suffix="de",
                                split="train",
                                tokens_in_batch=tokens_in_batch)

val_dataset = SequenceDataset(data_folder=data_folder,
                              source_suffix="en",
                              target_suffix="de",
                              split="val",
                              tokens_in_batch=tokens_in_batch)

model = Transformer(vocab_size=train_dataset.bpe_model.vocab_size(),
                    positional_encoding=positional_encoding,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_queries=d_queries,
                    d_values=d_values,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    dropout=dropout)

optimizer = nn.Adam(params=model.trainable_params(),
                    learning_rate=lr, betas=betas, eps=epsilon)
criterion = LabelSmoothedCE(eps=label_smoothing)
accumulator = Accumulator(optimizer, accumulate_step)

epochs = (n_steps // (train_dataset.n_batches // accumulate_step)) + 1

def forward_fn(source_seq, target_seq, source_seq_length, target_seq_lengths):
    """forward fn"""
    predicted_seq = model(source_seq, target_seq, source_seq_length, target_seq_lengths)
    loss =  criterion(predicted_seq, target_seq[:, 1:], target_seq_lengths - 1)

    return loss / accumulate_step

grad_fn = mindspore.value_and_grad(forward_fn, None, model.trainable_params())

def train_step(source_seq, target_seq, source_seq_length, target_seq_lengths, step):
    """train step"""
    loss, grads = grad_fn(source_seq, target_seq, source_seq_length, target_seq_lengths)
    accumulator(grads, step, d_model, warmup_steps)

    return loss, accumulator.counter

def train(epoch, step):
    """train"""
    model.set_train()

    # Track some metrics
    data_time = AverageMeter()  # data loading time
    step_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss

    start_data_time = time.time()
    start_step_time = time.time()

    for source_seq, target_seq, source_seq_length, target_seq_length in train_dataset:
        data_time.update(time.time() - start_data_time)
        loss, counter = train_step(source_seq, target_seq, source_seq_length, target_seq_length, step)

        losses.update(float(loss), int((target_seq_length-1).sum()))

        if counter % accumulate_step == 0:
            step += 1
            step_time.update(time.time() - start_step_time)

            if step % print_frequency == 0:
                print(f'Epoch {epoch+1}/{epochs}-----'
                      f'Batch {counter}/{train_dataset.n_batches}-----'
                      f'Step {step}/{n_steps}-----'
                      f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})-----'
                      f'Step Time {step_time.val:.3f} ({step_time.avg:.3f})-----'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})')

            start_step_time = time.time()

            if epoch in [epochs - 1, epochs - 2] and step % 1500 == 0:
                save_checkpoint(model, f'step{step}_transformer_epoch_{epoch}.ckpt')

        start_data_time = time.time()

def validate():
    """validate"""
    model.set_train(False)
    losses = AverageMeter()
    for source_seq, target_seq, source_seq_length, target_seq_length in tqdm(val_dataset, total=val_dataset.n_batches):
        predicted_seq = model(source_seq, target_seq, source_seq_length, target_seq_length)

        loss = criterion(predicted_seq, target_seq[:, 1:], target_seq_length - 1)
        losses.update(float(loss), int((target_seq_length - 1).sum()))

    print(f"\nValidation loss: {losses.avg:.3f}\n\n")

def main():
    """main"""
    for epoch in range(start_epoch, epochs):
        # step
        step = epoch * train_dataset.n_batches // accumulate_step

        train_dataset.create_batches()
        val_dataset.create_batches()
        train(epoch, step)
        validate()

        save_checkpoint(model, f'transformer_epoch_{epoch}.ckpt')

if __name__ == '__main__':
    main()

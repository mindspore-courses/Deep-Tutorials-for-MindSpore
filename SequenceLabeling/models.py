# pylint: disable=C0103
# pylint: disable=E0401
"""models"""

import mindspore
from mindspore import nn, ops
from utils import log_sum_exp

class Highway(nn.Cell):
    """
    Highway Network.
    """

    def __init__(self, size, num_layers=1, dropout=0.5):
        """
        :param size: size of linear layer (matches input size)
        :param num_layers: number of transform and gate layers
        :param dropout: dropout
        """
        super().__init__()
        self.size = size
        self.num_layers = num_layers
        self.transform = nn.CellList()  # list of transform layers
        self.gate = nn.CellList()  # list of gate layers
        self.dropout = nn.Dropout(p=dropout)

        for _ in range(num_layers):
            transform = nn.Dense(size, size)
            gate = nn.Dense(size, size)
            self.transform.append(transform)
            self.gate.append(gate)

    def construct(self, x):
        """
        Forward propagation.

        :param x: input tensor
        :return: output tensor, with same dimensions as input tensor
        """
        transformed = ops.relu(self.transform[0](x))  # transform input
        g = ops.sigmoid(self.gate[0](x))  # calculate how much of the transformed input to keep

        out = g * transformed + (1 - g) * x  # combine input and transformed input in this ratio

        # If there are additional layers
        for i in range(1, self.num_layers):
            out = self.dropout(out)
            transformed = ops.relu(self.transform[i](out))
            g = ops.sigmoid(self.gate[i](out))

            out = g * transformed + (1 - g) * out

        return out

class CRF(nn.Cell):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size):
        """
        :param hidden_dim: size of word RNN/BLSTM's output
        :param tagset_size: number of tags
        """
        super().__init__()
        self.batch_size = None
        self.timesteps = None
        self.tagset_size = tagset_size
        self.emission = nn.Dense(hidden_dim, self.tagset_size)
        self.transition = mindspore.Parameter(ops.zeros((self.tagset_size, self.tagset_size)))

    def construct(self, feats):
        """
        Forward propagation.

        :param feats: output of word RNN/BLSTM, a tensor of dimensions (batch_size, timesteps, hidden_dim)
        :return: CRF scores, a tensor of dimensions (batch_size, timesteps, tagset_size, tagset_size)
        """
        self.batch_size = feats.shape[0]
        self.timesteps = feats.shape[1]

        emission_scores = self.emission(feats)  # (batch_size, timesteps, tagset_size)
        emission_scores = emission_scores.unsqueeze(2).broadcast_to((self.batch_size, self.timesteps, self.tagset_size,
                                                              self.tagset_size))  # (batch_size, timesteps, tagset_size, tagset_size)

        crf_scores = emission_scores + self.transition.unsqueeze(0).unsqueeze(0)  # (batch_size, timesteps, tagset_size, tagset_size)
        return crf_scores

class LM_LSTM_CRF(nn.Cell):
    """
    The encompassing LM-LSTM-CRF model.
    """

    def __init__(self, tagset_size, charset_size, char_emb_dim, char_rnn_dim, char_rnn_layers, vocab_size,
                 lm_vocab_size, word_emb_dim, word_rnn_dim, word_rnn_layers, dropout, highway_layers=1):
        """
        :param tagset_size: number of tags
        :param charset_size: size of character vocabulary
        :param char_emb_dim: size of character embeddings
        :param char_rnn_dim: size of character RNNs/LSTMs
        :param char_rnn_layers: number of layers in character RNNs/LSTMs
        :param vocab_size: input vocabulary size
        :param lm_vocab_size: vocabulary size of language models (in-corpus words subject to word frequency threshold)
        :param word_emb_dim: size of word embeddings
        :param word_rnn_dim: size of word RNN/BLSTM
        :param word_rnn_layers:  number of layers in word RNNs/LSTMs
        :param dropout: dropout
        :param highway_layers: number of transform and gate layers
        """

        super().__init__()

        self.batch_size = None
        self.word_pad_len = None
        self.tagset_size = tagset_size  # this is the size of the output vocab of the tagging model

        self.charset_size = charset_size
        self.char_emb_dim = char_emb_dim
        self.char_rnn_dim = char_rnn_dim
        self.char_rnn_layers = char_rnn_layers

        self.wordset_size = vocab_size  # this is the size of the input vocab (embedding layer) of the tagging model
        self.lm_vocab_size = lm_vocab_size  # this is the size of the output vocab of the language model
        self.word_emb_dim = word_emb_dim
        self.word_rnn_dim = word_rnn_dim
        self.word_rnn_layers = word_rnn_layers

        self.highway_layers = highway_layers

        self.dropout = nn.Dropout(p=dropout)

        self.char_embeds = nn.Embedding(self.charset_size, self.char_emb_dim)  # character embedding layer
        self.forw_char_lstm = nn.LSTM(self.char_emb_dim, self.char_rnn_dim, num_layers=self.char_rnn_layers,
                                      bidirectional=False, dropout=dropout, batch_first=True)  # forward character LSTM
        self.back_char_lstm = nn.LSTM(self.char_emb_dim, self.char_rnn_dim, num_layers=self.char_rnn_layers,
                                      bidirectional=False, dropout=dropout, batch_first=True)  # backward character LSTM

        self.word_embeds = nn.Embedding(self.wordset_size, self.word_emb_dim)  # word embedding layer
        self.word_blstm = nn.LSTM(self.word_emb_dim + self.char_rnn_dim * 2, self.word_rnn_dim // 2,
                                  num_layers=self.word_rnn_layers, bidirectional=True, dropout=dropout,
                                  batch_first=True)  # word BLSTM

        self.crf = CRF((self.word_rnn_dim // 2) * 2, self.tagset_size)  # conditional random field

        self.forw_lm_hw = Highway(self.char_rnn_dim, num_layers=self.highway_layers,
                                  dropout=dropout)  # highway to transform forward char LSTM output for the forward language model
        self.back_lm_hw = Highway(self.char_rnn_dim, num_layers=self.highway_layers,
                                  dropout=dropout)  # highway to transform backward char LSTM output for the backward language model
        self.subword_hw = Highway(2 * self.char_rnn_dim, num_layers=self.highway_layers,
                                  dropout=dropout)  # highway to transform combined forward and backward char LSTM outputs for use in the word BLSTM

        self.forw_lm_out = nn.Dense(self.char_rnn_dim,
                                     self.lm_vocab_size)  # linear layer to find vocabulary scores for the forward language model
        self.back_lm_out = nn.Dense(self.char_rnn_dim,
                                     self.lm_vocab_size)  # linear layer to find vocabulary scores for the backward language model

    def init_word_embeddings(self, embeddings):
        """
        Initialize embeddings with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.word_embeds.embedding_table = mindspore.Parameter(embeddings)

    def fine_tune_word_embeddings(self, fine_tune=False):
        """
        Fine-tune embedding layer? (Not fine-tuning only makes sense if using pre-trained embeddings).

        :param fine_tune: Fine-tune?
        """
        for p in self.word_embeds.get_parameters():
            p.requires_grad = fine_tune

    def construct(self, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps, tmaps, wmap_lengths, cmap_lengths):
        """
        Forward propagation.

        :param cmaps_f: padded encoded forward character sequences, a tensor of dimensions (batch_size, char_pad_len)
        :param cmaps_b: padded encoded backward character sequences, a tensor of dimensions (batch_size, char_pad_len)
        :param cmarkers_f: padded forward character markers, a tensor of dimensions (batch_size, word_pad_len)
        :param cmarkers_b: padded backward character markers, a tensor of dimensions (batch_size, word_pad_len)
        :param wmaps: padded encoded word sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param tmaps: padded tag sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param wmap_lengths: word sequence lengths, a tensor of dimensions (batch_size)
        :param cmap_lengths: character sequence lengths, a tensor of dimensions (batch_size, word_pad_len)
        """
        self.batch_size = cmaps_f.shape[0]
        self.word_pad_len = wmaps.shape[1]

        # Sort by decreasing true char. sequence length
        cmap_lengths, char_sort_ind = cmap_lengths.sort(axis=0, descending=True)
        cmaps_f = cmaps_f[char_sort_ind]
        cmaps_b = cmaps_b[char_sort_ind]
        cmarkers_f = cmarkers_f[char_sort_ind]
        cmarkers_b = cmarkers_b[char_sort_ind]
        wmaps = wmaps[char_sort_ind]
        tmaps = tmaps[char_sort_ind]
        wmap_lengths = wmap_lengths[char_sort_ind]

        # Embedding look-up for characters
        cf = self.char_embeds(cmaps_f)  # (batch_size, char_pad_len, char_emb_dim)
        cb = self.char_embeds(cmaps_b)

        # Dropout
        cf = self.dropout(cf)  # (batch_size, char_pad_len, char_emb_dim)
        cb = self.dropout(cb)

        # LSTM
        cf, _ = self.forw_char_lstm(cf, seq_length=cmap_lengths)  # packed sequence of char_rnn_dim, with real sequence lengths
        cb, _ = self.back_char_lstm(cb, seq_length=cmap_lengths)

        # Sanity check
        assert cf.shape[1] == max(cmap_lengths.asnumpy().tolist()) == list(cmap_lengths)[0]

        # Select RNN outputs only at marker points (spaces in the character sequence)
        cmarkers_f = cmarkers_f.unsqueeze(2).broadcast_to((self.batch_size, self.word_pad_len, self.char_rnn_dim))
        cmarkers_b = cmarkers_b.unsqueeze(2).broadcast_to((self.batch_size, self.word_pad_len, self.char_rnn_dim))
        cf_selected = ops.gather_elements(cf, 1, cmarkers_f)  # (batch_size, word_pad_len, char_rnn_dim)
        cb_selected = ops.gather_elements(cb, 1, cmarkers_b)

        # Only for co-training, not useful for tagging after model is trained
        if self.training:
            lm_f = self.forw_lm_hw(self.dropout(cf_selected))  # (batch_size, word_pad_len, char_rnn_dim)
            lm_b = self.back_lm_hw(self.dropout(cb_selected))
            lm_f_scores = self.forw_lm_out(self.dropout(lm_f))  # (batch_size, word_pad_len, lm_vocab_size)
            lm_b_scores = self.back_lm_out(self.dropout(lm_b))

        # Sort by decreasing true word sequence length
        wmap_lengths, word_sort_ind = wmap_lengths.sort(axis=0, descending=True)
        wmaps = wmaps[word_sort_ind]
        tmaps = tmaps[word_sort_ind]
        cf_selected = cf_selected[word_sort_ind]  # for language model
        cb_selected = cb_selected[word_sort_ind]
        if self.training:
            lm_f_scores = lm_f_scores[word_sort_ind]
            lm_b_scores = lm_b_scores[word_sort_ind]

        # Embedding look-up for words
        w = self.word_embeds(wmaps)  # (batch_size, word_pad_len, word_emb_dim)
        w = self.dropout(w)

        # Sub-word information at each word
        subword = self.subword_hw(self.dropout(
            ops.cat((cf_selected, cb_selected), axis=2)))  # (batch_size, word_pad_len, 2 * char_rnn_dim)
        subword = self.dropout(subword)

        # Concatenate word embeddings and sub-word features
        w = ops.cat((w, subword), axis=2)  # (batch_size, word_pad_len, word_emb_dim + 2 * char_rnn_dim)

        # LSTM
        w, _ = self.word_blstm(w, seq_length=wmap_lengths)  # packed sequence of word_rnn_dim, with real sequence lengths

        w = self.dropout(w)

        crf_scores = self.crf(w)  # (batch_size, max_word_len_in_batch, tagset_size, tagset_size)

        if self.training:
            return crf_scores, lm_f_scores, lm_b_scores, wmaps, tmaps, wmap_lengths, word_sort_ind, char_sort_ind

        return crf_scores, wmaps, tmaps, wmap_lengths, word_sort_ind, char_sort_ind

class ViterbiLoss(nn.Cell):
    """
    Viterbi Loss.
    """

    def __init__(self, tag_map):
        """
        :param tag_map: tag map
        """
        super().__init__()
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def construct(self, scores, targets, lengths):
        """
        Forward propagation.

        :param scores: CRF scores
        :param targets: true tags indices in unrolled CRF scores
        :param lengths: word sequence lengths
        :return: viterbi loss
        """

        batch_size = scores.shape[0]
        word_pad_len = scores.shape[1]

        # Gold score

        targets = targets.unsqueeze(2)
        scores_at_targets = ops.gather_elements(scores.view(batch_size, word_pad_len, -1), 2, targets).squeeze(2)

        gold_score = scores_at_targets.sum()

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = ops.zeros((batch_size, self.tagset_size))

        for t in range(max(lengths)):
            batch_size_t = sum(1 for l in lengths if l > t)  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and log-sum-exp
                # Remember, the cur_tag of the previous timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores along cur. timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = log_sum_exp(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    axis=1)  # (batch_size, tagset_size)

        # We only need the final accumulated scores at the <end> tag
        all_paths_scores = scores_upto_t[:, self.end_tag].sum()

        viterbi_loss = all_paths_scores - gold_score
        viterbi_loss = viterbi_loss / batch_size

        return viterbi_loss

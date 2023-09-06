# pylint: disable=C0103
"""model HierarchialAttentionNetwork"""

import mindspore
from mindspore import nn, ops


class HierarchialAttentionNetwork(nn.Cell):
    """
    The overarching Hierarchial Attention Network (HAN).
    """

    def __init__(self, n_classes, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers,
                 sentence_rnn_layers, word_att_size, sentence_att_size, dropout=0.5):
        """
        :param n_classes: number of classes
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param sentence_rnn_size: size of (bidirectional) sentence-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param sentence_rnn_layers: number of layers in sentence-level RNN
        :param word_att_size: size of word-level attention layer
        :param sentence_att_size: size of sentence-level attention layer
        :param dropout: dropout
        """
        super().__init__()

        # Sentence-level attention module (which will, in-turn, contain the word-level attention module)
        self.sentence_attention = SentenceAttention(vocab_size, emb_size, word_rnn_size, sentence_rnn_size,
                                                    word_rnn_layers, sentence_rnn_layers, word_att_size,
                                                    sentence_att_size, dropout)

        # Classifier
        self.fc = nn.Dense(2 * sentence_rnn_size, n_classes)

        self.dropout = nn.Dropout(p=dropout)

    def construct(self, documents, sentences_per_document, words_per_sentence):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: class scores, attention weights of words, attention weights of sentences
        """
        # Apply sentence-level attention module (and in turn, word-level attention module) to get document embeddings
        document_embeddings, word_alphas, sentence_alphas = self.sentence_attention(documents, sentences_per_document,
                                                                                    words_per_sentence)

        # Classify
        scores = self.fc(self.dropout(document_embeddings))  # (n_documents, n_classes)

        return scores, word_alphas, sentence_alphas


class SentenceAttention(nn.Cell):
    """
    The sentence-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers, sentence_rnn_layers,
                 word_att_size, sentence_att_size, dropout):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param sentence_rnn_size: size of (bidirectional) sentence-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param sentence_rnn_layers: number of layers in sentence-level RNN
        :param word_att_size: size of word-level attention layer
        :param sentence_att_size: size of sentence-level attention layer
        :param dropout: dropout
        """
        super().__init__()

        # Word-level attention module
        self.word_attention = WordAttention(vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size,
                                            dropout)

        # Bidirectional sentence-level RNN
        self.sentence_rnn = nn.GRU(2 * word_rnn_size, sentence_rnn_size, num_layers=sentence_rnn_layers,
                                   bidirectional=True, dropout=dropout, batch_first=True)

        # Sentence-level attention network
        self.sentence_attention = nn.Dense(2 * sentence_rnn_size, sentence_att_size)

        # Sentence context vector to take dot-product with
        self.sentence_context_vector = nn.Dense(sentence_att_size, 1, has_bias=False)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, documents, sentences_per_document, words_per_sentence):
        """
        Forward propagation.

        :param documents: encoded document-level data, a tensor of dimensions (n_documents, sent_pad_len, word_pad_len)
        :param sentences_per_document: document lengths, a tensor of dimensions (n_documents)
        :param words_per_sentence: sentence lengths, a tensor of dimensions (n_documents, sent_pad_len)
        :return: document embeddings, attention weights of words, attention weights of sentences
        """

        # Find sentence embeddings by applying the word-level attention module
        sentences, word_alphas = self.word_attention(ops.flatten(documents, start_dim=0, end_dim=1),
                                                     ops.flatten(words_per_sentence, start_dim=0, end_dim=1))
        # (n_documents*sent_pad_len, 2*word_rnn_size), (n_documents*sent_pad_len, word_pad_len)

        sentences = self.dropout(sentences)

        # Apply the sentence-level RNN over the sentence embeddings
        # the last dimension of the input tensor should be 2*word_rnn_size
        documents, _ = self.sentence_rnn(ops.reshape(sentences, (sentences_per_document.shape[0], -1, sentences.shape[-1])),
                                         seq_length=sentences_per_document)
        # the output of the RNN (n_documents, sent_pad_len, 2 * sentence_rnn_size)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_s = self.sentence_attention(documents)  # (n_documents, sent_pad_len, att_size)
        att_s = ops.tanh(att_s)  # (n_documents, sent_pad_len, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_s = self.sentence_context_vector(att_s).squeeze(2)  # (n_documents, sent_pad_len)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over sentences in the same document

        # First, take the exponent
        max_value = att_s.max()  # scalar, for numerical stability during exponent calculation
        att_s = ops.exp(att_s - max_value)  # (n_documents, sent_pad_len)


        # Calculate softmax values as now sentences are arranged in their respective documents
        sentence_alphas = att_s / ops.sum(att_s, dim=1, keepdim=True)  # (n_documents, sent_pad_len)

        # Find document embeddings
        documents = documents * sentence_alphas.unsqueeze(2)
        # (n_documents, sent_pad_len, 2 * sentence_rnn_size)

        documents = documents.sum(axis=1)  # (n_documents, 2 * sentence_rnn_size)

        # Also re-arrange word_alphas (SENTENCES -> DOCUMENTS)
        word_alphas = ops.reshape(word_alphas, (sentences_per_document.shape[0], -1, word_alphas.shape[-1]))
        # (n_documents, sent_pad_len, word_pad_len)

        return documents, word_alphas, sentence_alphas


class WordAttention(nn.Cell):
    """
    The word-level attention module.
    """

    def __init__(self, vocab_size, emb_size, word_rnn_size, word_rnn_layers, word_att_size, dropout):
        """
        :param vocab_size: number of words in the vocabulary of the model
        :param emb_size: size of word embeddings
        :param word_rnn_size: size of (bidirectional) word-level RNN
        :param word_rnn_layers: number of layers in word-level RNN
        :param word_att_size: size of word-level attention layer
        :param dropout: dropout
        """
        super().__init__()

        # Embeddings (look-up) layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)

        # Bidirectional word-level RNN
        self.word_rnn = nn.GRU(emb_size, word_rnn_size, num_layers=word_rnn_layers, bidirectional=True,
                               dropout=dropout, batch_first=True)

        # Word-level attention network
        self.word_attention = nn.Dense(2 * word_rnn_size, word_att_size)

        # Word context vector to take dot-product with
        self.word_context_vector = nn.Dense(word_att_size, 1, has_bias=False)
        # You could also do this with:
        # self.word_context_vector = nn.Parameter(torch.FloatTensor(1, word_att_size))
        # self.word_context_vector.data.uniform_(-0.1, 0.1)
        # And then take the dot-product

        self.dropout = nn.Dropout(p=dropout)

    def init_embeddings(self, embeddings):
        """
        Initialized embedding layer with pre-computed embeddings.

        :param embeddings: pre-computed embeddings
        """
        self.embeddings.embedding_table = mindspore.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=False):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: allow?
        """
        for p in self.embeddings.get_parameters():
            p.requires_grad = fine_tune

    def construct(self, sentences, words_per_sentence):
        """
        Forward propagation.

        :param sentences: encoded sentence-level data, a tensor of dimension (n_sentences, word_pad_len, emb_size)
        :param words_per_sentence: sentence lengths, a tensor of dimension (n_sentences)
        :return: sentence embeddings, attention weights of words
        """

        # Get word embeddings, apply dropout
        sentences = self.dropout(self.embeddings(sentences))  # (n_documents*sent_pad_len, word_pad_len, emb_size)

        # Apply the word-level RNN over the word embeddings
        sentences, _ = self.word_rnn(sentences, seq_length=words_per_sentence)
        #the output of the RNN (n_documents*sent_pad_len, word_pad_len, 2 * word_rnn_size)

        # Find attention vectors by applying the attention linear layer on the output of the RNN
        att_w = self.word_attention(sentences)  # (n_documents*sent_pad_len, word_pad_len, att_size)
        att_w = ops.tanh(att_w)  # (n_documents*sent_pad_len, word_pad_len, att_size)
        # Take the dot-product of the attention vectors with the context vector (i.e. parameter of linear layer)
        att_w = self.word_context_vector(att_w).squeeze(2)  # (n_documents*sent_pad_len, word_pad_len)

        # Compute softmax over the dot-product manually
        # Manually because they have to be computed only over words in the same sentence

        # First, take the exponent
        max_value = att_w.max()  # scalar, for numerical stability during exponent calculation
        att_w = ops.exp(att_w - max_value)  # (n_documents*sent_pad_len, word_pad_len)

        # Calculate softmax values as now words are arranged in their respective sentences
        word_alphas = att_w / ops.sum(att_w, dim=1, keepdim=True)  # (n_documents*sent_pad_len, word_pad_len)

        # Find sentence embeddings
        sentences = sentences * word_alphas.unsqueeze(2)  # (n_documents*sent_pad_len, word_pad_len, 2 * word_rnn_size)
        sentences = sentences.sum(axis=1)  # (n_documents*sent_pad_len, 2 * word_rnn_size)

        return sentences, word_alphas

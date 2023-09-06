# pylint: disable=C0103
"""classify the documents"""

import json
import os
import mindspore
from mindspore import ops, load_checkpoint, load_param_into_net
from model import HierarchialAttentionNetwork
from utils import preprocess, rev_label_map, label_map, load_word2vec_embeddings
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from PIL import Image, ImageDraw, ImageFont

# Evaluation parameters
batch_size = 64  # batch size
workers = 4  # number of workers
print_freq = 2000  # print training or validation status every __ batches
checkpoint = 'BEST_checkpoint_han.ckpt'

data_folder = '/han data'
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

# Load model
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

# Pad limits, can use any high-enough value since our model does not compute over the pads
sentence_limit = 15
word_limit = 20

# Tokenizers
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()


def classify(document):
    """
    Classify a document with the Hierarchial Attention Network (HAN).

    :param document: a document in text form
    :return: pre-processed tokenized document, class scores, attention weights for words, attention weights for sentences, sentence lengths
    """
    # A list to store the document tokenized into words
    doc = []

    # Tokenize document into sentences
    sentences = []
    for paragraph in preprocess(document).splitlines():
        sentences.extend(list(sent_tokenizer.tokenize(paragraph)))

    # Tokenize sentences into words
    for s in sentences[:sentence_limit]:
        w = word_tokenizer.tokenize(s)[:word_limit]
        if len(w) == 0:
            continue
        doc.append(w)

    # Number of sentences in the document
    sentences_in_doc = len(doc)
    sentences_in_doc = mindspore.Tensor([sentences_in_doc], dtype=mindspore.int64)

    # Number of words in each sentence
    words_in_each_sentence = list(map(len, doc))
    words_in_each_sentence = mindspore.Tensor(words_in_each_sentence, dtype=mindspore.int64).unsqueeze(0)  # (1, n_sentences)

    # Encode document with indices from the word map
    encoded_doc = list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc))
    encoded_doc = mindspore.Tensor(encoded_doc, dtype=mindspore.int64).unsqueeze(0)

    # Apply the HAN model
    scores, word_alphas, sentence_alphas = model(encoded_doc, sentences_in_doc, words_in_each_sentence)
    scores = scores.squeeze(0)  # (n_classes)
    scores = ops.softmax(scores, axis=0)  # (n_classes)
    word_alphas = word_alphas.squeeze(0)  # (n_sentences, max_sent_len_in_document)
    sentence_alphas = sentence_alphas.squeeze(0)  # (n_sentences)
    words_in_each_sentence = words_in_each_sentence.squeeze(0)  # (n_sentences)

    return doc, scores, word_alphas, sentence_alphas, words_in_each_sentence


def visualize_attention(doc, scores, word_alphas, sentence_alphas, words_in_each_sentence):
    """
    Visualize important sentences and words, as seen by the HAN model.

    :param doc: pre-processed tokenized document
    :param scores: class scores, a tensor of size (n_classes)
    :param word_alphas: attention weights of words, a tensor of size (n_sentences, max_sent_len_in_document)
    :param sentence_alphas: attention weights of sentences, a tensor of size (n_sentences)
    :param words_in_each_sentence: sentence lengths, a tensor of size (n_sentences)
    """
    # Find best prediction
    score, prediction = scores.max(axis=0)
    prediction = f'{rev_label_map[int(prediction)]} ({float(score*100):.2f}'

    # For each word, find it's effective importance (sentence alpha * word alpha)
    alphas = (sentence_alphas.unsqueeze(1) * word_alphas * words_in_each_sentence.unsqueeze(
        1).float() / words_in_each_sentence.max().float())

    # Determine size of the image, visualization properties for each word, and each sentence
    min_font_size = 15  # minimum size possible for a word, because size is scaled by normalized word*sentence alphas
    max_font_size = 55  # maximum size possible for a word, because size is scaled by normalized word*sentence alphas
    space_size = ImageFont.truetype("./calibril.ttf", max_font_size).getsize(' ')  # use spaces of maximum font size
    line_spacing = 15  # spacing between sentences
    left_buffer = 100  # initial empty space on the left where sentence-rectangles will be drawn
    top_buffer = 2 * min_font_size + 3 * line_spacing  # initial empty space on the top where the detected category will be displayed
    image_width = left_buffer  # width of the entire image so far
    image_height = top_buffer + line_spacing  # height of the entire image so far
    word_loc = [image_width, image_height]  # top-left coordinates of the next word that will be printed
    rectangle_height = 0.75 * max_font_size  # height of the rectangles that will represent sentence alphas
    max_rectangle_width = 0.8 * left_buffer  # maximum width of the rectangles that will represent sentence alphas, scaled by sentence alpha
    rectangle_loc = [0.9 * left_buffer,
                     image_height + rectangle_height]  # bottom-right coordinates of next rectangle that will be printed
    word_viz_properties = []
    sentence_viz_properties = []
    for s, sentence in enumerate(doc):
        # Find visualization properties for each sentence, represented by rectangles
        # Factor to scale by
        sentence_factor = float(sentence_alphas[s]) / float(sentence_alphas.max())

        # Color of rectangle
        rectangle_saturation = str(int(sentence_factor * 100))
        rectangle_lightness = str(25 + 50 - int(sentence_factor * 50))
        rectangle_color = 'hsl(0,' + rectangle_saturation + '%,' + rectangle_lightness + '%)'

        # Bounds of rectangle
        rectangle_bounds = [rectangle_loc[0] - sentence_factor * max_rectangle_width,
                            rectangle_loc[1] - rectangle_height] + rectangle_loc

        # Save sentence's rectangle's properties
        sentence_viz_properties.append({'bounds': rectangle_bounds.copy(),
                                        'color': rectangle_color})

        for w, word in enumerate(sentence):
            # Find visualization properties for each word
            # Factor to scale by
            word_factor = float(alphas[s, w]) / float(alphas.max())

            # Color of word
            word_saturation = str(int(word_factor * 100))
            word_lightness = str(25 + 50 - int(word_factor * 50))
            word_color = 'hsl(0,' + word_saturation + '%,' + word_lightness + '%)'

            # Size of word
            word_font_size = int(min_font_size + word_factor * (max_font_size - min_font_size))
            word_font = ImageFont.truetype("./calibril.ttf", word_font_size)

            # Save word's properties
            word_viz_properties.append({'loc': word_loc.copy(),
                                        'word': word,
                                        'font': word_font,
                                        'color': word_color})

            # Update word and sentence locations for next word, height, width values
            word_size = word_font.getsize(word)
            word_loc[0] += word_size[0] + space_size[0]
            image_width = max(image_width, word_loc[0])
        word_loc[0] = left_buffer
        word_loc[1] += max_font_size + line_spacing
        image_height = max(image_height, word_loc[1])
        rectangle_loc[1] += max_font_size + line_spacing

    # Create blank image
    img = Image.new('RGB', (image_width, image_height), (255, 255, 255))

    # Draw
    draw = ImageDraw.Draw(img)
    # Words
    for viz in word_viz_properties:
        draw.text(xy=viz['loc'], text=viz['word'], fill=viz['color'], font=viz['font'])
    # Rectangles that represent sentences
    for viz in sentence_viz_properties:
        draw.rectangle(xy=viz['bounds'], fill=viz['color'])
    # Detected category/topic
    category_font = ImageFont.truetype("./calibril.ttf", min_font_size)
    draw.text(xy=[line_spacing, line_spacing], text='Detected Category:', fill='grey', font=category_font)
    draw.text(xy=[line_spacing, line_spacing + category_font.getsize('Detected Category:')[1] + line_spacing],
              text=prediction.upper(), fill='black',
              font=category_font)
    del draw

    # Display
    img.show()


if __name__ == '__main__':
    test_document = 'How do computers work? I have a CPU I want to use. But my keyboard and motherboard do not help.\
                    \n\n You can just google how computers work. Honestly, its easy.'
    test_document = 'But think about it! It\'s so cool. Physics is really all about math. what feynman said, hehe'

    test_document = "I think I'm falling sick. There was some indigestion at first. But now a fever is beginning to take hold."

    test_document = "I want to tell you something important. Get into the stock market and investment funds. \
                    Make some money so you can buy yourself some yogurt."
    test_document = "You know what's wrong with this country? republicans and democrats. always at each other's throats\
                    \n There's no respect, no bipartisanship."
    visualize_attention(*classify(test_document))

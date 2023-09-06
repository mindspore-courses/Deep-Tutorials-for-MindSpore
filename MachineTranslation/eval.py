# pylint: disable=C0103
"""evaluate the model"""

import codecs
import os
import sacrebleu
from translate import translate
from tqdm import tqdm
from data_generator import SequenceDataset

# Use sacreBLEU in Python or in the command-line?
# Using in Python will use the test data downloaded in prepare_data.py
# Using in the command-line will use test data automatically downloaded by sacreBLEU...
# ...and will print a standard signature which represents the exact BLEU method used! (Important for others to be able to reproduce or compare!)
sacrebleu_in_python = False

# Make sure the right model checkpoint is selected in translate.py

# Data loader
test_dataset = SequenceDataset(data_folder="/transformer data",
                             source_suffix="en",
                             target_suffix="de",
                             split="test",
                             tokens_in_batch=None)
test_dataset.create_batches()

# Evaluate
hypotheses = []
references = []
for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(
        tqdm(test_dataset, total=test_dataset.n_batches)):
    hypotheses.append(translate(source_sequence=source_sequence,
                                beam_size=4,
                                length_norm_coefficient=0.6)[0])
    references.extend(test_dataset.bpe_model.decode(target_sequence.asnumpy().tolist(), ignore_ids=[0, 2, 3]))
if sacrebleu_in_python:
    print("\n13a tokenization, cased:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references]))
    print("\n13a tokenization, caseless:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True))
    print("\nInternational tokenization, cased:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl'))
    print("\nInternational tokenization, caseless:\n")
    print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl', lowercase=True))
    print("\n")
else:
    with codecs.open("translated_test.de", "w", encoding="utf-8") as f:
        f.write("\n".join(hypotheses))
    print("\n13a tokenization, cased:\n")
    os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de")
    print("\n13a tokenization, caseless:\n")
    os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -lc")
    print("\nInternational tokenization, cased:\n")
    os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -tok intl")
    print("\nInternational tokenization, caseless:\n")
    os.system("cat translated_test.de | sacrebleu -t wmt14/full -l en-de -tok intl -lc")
    print("\n")

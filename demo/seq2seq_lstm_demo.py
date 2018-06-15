#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os
import pickle
import sys
import random

try:
    from seq2seq_lstm import Seq2SeqLSTM
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from seq2seq_lstm import Seq2SeqLSTM


def load_text_pairs(file_name):
    input_texts = list()
    target_texts = list()
    line_idx = 1
    with codecs.open(file_name, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = u'File "{0}": line {1} is wrong!'.format(file_name, line_idx)
                line_parts = prep_line.split(u'\t')
                assert len(line_parts) == 2, err_msg
                new_input_text = line_parts[0].strip()
                new_target_text = line_parts[1].strip()
                assert (len(new_input_text) > 0) and (len(new_target_text) > 0), err_msg
                input_texts.append(tokenize_text(new_input_text))
                target_texts.append(tokenize_text(new_target_text))
            cur_line = fp.readline()
            line_idx += 1
    return input_texts, target_texts


def shuffle_text_pairs(*args):
    assert len(args) == 2, u'Text pairs (input and target texts) are specified incorrectly!'
    indices = list(range(len(args[0])))
    random.shuffle(indices)
    input_texts = []
    target_texts = []
    for ind in indices:
        input_texts.append(args[0][ind])
        target_texts.append(args[1][ind])
    return input_texts, target_texts


def tokenize_text(src):
    tokens = list()
    for cur in src.split():
        tokens += list(cur)
        tokens.append(u'<space>')
    return u' '.join(tokens[:-1])

def detokenize_text(src):
    new_text = u''
    for cur_token in src.split():
        if cur_token == u'<space>':
            new_text += u' '
        else:
            new_text += cur_token
    return new_text.strip()

def estimate(predicted_texts, true_texts):
    n_err = 0
    n_total = len(predicted_texts)
    for i in range(n_total):
        pred_ = detokenize_text(predicted_texts[i]).lower()
        true_ = detokenize_text(true_texts[i]).lower()
        if pred_ != true_:
            n_err += 1
    return 1.0 - (n_err / float(n_total))


def main():
    if len(sys.argv) > 1:
        model_name = os.path.normpath(sys.argv[1].strip())
        if len(model_name) == 0:
            model_name = None
        else:
            model_dir_name = os.path.dirname(model_name)
            if len(model_dir_name) > 0:
                assert os.path.isdir(model_dir_name), u'Directory "{0}" does not exist!'.format(model_dir_name)
    else:
        model_name = None

    input_texts_for_training, target_texts_for_training = shuffle_text_pairs(
        *load_text_pairs(
            os.path.join(os.path.dirname(__file__), '..', 'data', 'eng_rus_for_training.txt')
        )
    )
    print(u'')
    print(u'There are {0} text pairs in the training data.'.format(len(input_texts_for_training)))
    print(u'Some samples of these text pairs:')
    for ind in range(10):
        input_text = input_texts_for_training[ind]
        target_text = target_texts_for_training[ind]
        print(u'    ' + detokenize_text(input_text) + u'\t' + detokenize_text(target_text))
    print(u'')

    input_texts_for_testing, target_texts_for_testing = load_text_pairs(
        os.path.join(os.path.dirname(__file__), '..', 'data', 'eng_rus_for_testing.txt')
    )
    print(u'There are {0} text pairs in the testing data.'.format(len(input_texts_for_testing)))
    print(u'Some samples of these text pairs:')
    indices = list(range(len(input_texts_for_testing)))
    random.shuffle(indices)
    for ind in indices[:10]:
        input_text = input_texts_for_testing[ind]
        target_text = target_texts_for_testing[ind]
        print(u'    ' + detokenize_text(input_text) + u'\t' + detokenize_text(target_text))
    print(u'')

    if (model_name is not None) and os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            seq2seq = pickle.load(fp)
        assert isinstance(seq2seq, Seq2SeqLSTM), \
            u'A sequence-to-sequence neural model cannot be loaded from file "{0}".'.format(model_name)
        print(u'')
        print(u'Model has been successfully loaded from file "{0}".'.format(model_name))
    else:
        seq2seq = Seq2SeqLSTM(latent_dim=512, validation_split=0.1, epochs=200, lr=1e-2, decay=1e-1, verbose=True,
                              lowercase=False)
        seq2seq.fit(input_texts_for_training, target_texts_for_training)
        print(u'')
        print(u'Training has been successfully finished.')
        if model_name is not None:
            with open(model_name, 'wb') as fp:
                pickle.dump(seq2seq, fp)
            print(u'Model has been successfully saved into file "{0}".'.format(model_name))

    predicted_texts = seq2seq.predict(input_texts_for_testing)
    sentence_correct = estimate(predicted_texts, target_texts_for_testing)
    print(u'')
    print(u'{0} texts have been predicted.'.format(len(predicted_texts)))
    print(u'Some samples of predicted text pairs:')
    for ind in indices[:10]:
        input_text = input_texts_for_testing[ind]
        target_text = predicted_texts[ind]
        print(u'    ' + detokenize_text(input_text) + u'\t' + detokenize_text(target_text))
    print(u'')
    print(u'Total sentence correct is {0:.2%}.'.format(sentence_correct))


if __name__ == '__main__':
    main()

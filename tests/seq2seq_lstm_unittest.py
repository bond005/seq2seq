# -*- coding: utf-8 -*-

import codecs
import os
import pickle
import re
import sys
import unittest

from keras import Model
import numpy as np
from sklearn.utils.validation import NotFittedError

try:
    from seq2seq_lstm import Seq2SeqLSTM
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from seq2seq_lstm import Seq2SeqLSTM


class TestSeq2SeqLSTM(unittest.TestCase):
    def setUp(self):
        self.training_set_name = os.path.join(os.path.dirname(__file__), 'testdata', 'eng_rus_for_training.txt')
        self.testing_set_name = os.path.join(os.path.dirname(__file__), 'testdata', 'eng_rus_for_testing.txt')
        self.model_name = os.path.join(os.path.dirname(__file__), 'testdata', 'seq2seq_lstm.pkl')

    def tearDown(self):
        if os.path.isfile(self.model_name):
            os.remove(self.model_name)

    def test_creation(self):
        seq2seq = Seq2SeqLSTM(batch_size=128, epochs=200, latent_dim=500, validation_split=0.1, decay=0.2, dropout=0.3,
                              recurrent_dropout=0.35, grad_clipping=50.0, lr=0.01, rho=0.8, epsilon=0.2,
                              lowercase=False, verbose=True)
        self.assertIsInstance(seq2seq, Seq2SeqLSTM)
        self.assertTrue(hasattr(seq2seq, 'batch_size'))
        self.assertEqual(seq2seq.batch_size, 128)
        self.assertTrue(hasattr(seq2seq, 'epochs'))
        self.assertEqual(seq2seq.epochs, 200)
        self.assertTrue(hasattr(seq2seq, 'latent_dim'))
        self.assertEqual(seq2seq.latent_dim, 500)
        self.assertTrue(hasattr(seq2seq, 'validation_split'))
        self.assertAlmostEqual(seq2seq.validation_split, 0.1)
        self.assertTrue(hasattr(seq2seq, 'decay'))
        self.assertAlmostEqual(seq2seq.decay, 0.2)
        self.assertTrue(hasattr(seq2seq, 'dropout'))
        self.assertAlmostEqual(seq2seq.dropout, 0.3)
        self.assertTrue(hasattr(seq2seq, 'recurrent_dropout'))
        self.assertAlmostEqual(seq2seq.recurrent_dropout, 0.35)
        self.assertTrue(hasattr(seq2seq, 'grad_clipping'))
        self.assertAlmostEqual(seq2seq.grad_clipping, 50.0)
        self.assertTrue(hasattr(seq2seq, 'lr'))
        self.assertAlmostEqual(seq2seq.lr, 0.01)
        self.assertTrue(hasattr(seq2seq, 'rho'))
        self.assertAlmostEqual(seq2seq.rho, 0.8)
        self.assertTrue(hasattr(seq2seq, 'lowercase'))
        self.assertFalse(seq2seq.lowercase)
        self.assertTrue(hasattr(seq2seq, 'verbose'))
        self.assertTrue(seq2seq.verbose)

    def test_fit_positive01(self):
        """ Input and target texts for training are the Python tuples. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM()
        res = seq2seq.fit(input_texts_for_training, target_texts_for_training)
        self.assertIsInstance(res, Seq2SeqLSTM)
        self.assertTrue(hasattr(res, 'input_token_index_'))
        self.assertIsInstance(res.input_token_index_, dict)
        self.assertTrue(hasattr(res, 'target_token_index_'))
        self.assertIsInstance(res.target_token_index_, dict)
        self.assertTrue(hasattr(res, 'reverse_target_char_index_'))
        self.assertIsInstance(res.reverse_target_char_index_, dict)
        self.assertTrue(hasattr(res, 'max_encoder_seq_length_'))
        self.assertIsInstance(res.max_encoder_seq_length_, int)
        self.assertGreater(res.max_encoder_seq_length_, 0)
        self.assertTrue(hasattr(res, 'max_decoder_seq_length_'))
        self.assertIsInstance(res.max_decoder_seq_length_, int)
        self.assertGreater(res.max_decoder_seq_length_, 0)
        self.assertTrue(hasattr(res, 'encoder_model_'))
        self.assertIsInstance(res.encoder_model_, Model)
        self.assertTrue(hasattr(res, 'encoder_model_'))
        self.assertIsInstance(res.decoder_model_, Model)

    def test_fit_positive02(self):
        """ Input and target texts for training are the 1-D numpy arrays. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM()
        res = seq2seq.fit(np.array(input_texts_for_training), np.array(target_texts_for_training))
        self.assertIsInstance(res, Seq2SeqLSTM)
        self.assertTrue(hasattr(res, 'input_token_index_'))
        self.assertIsInstance(res.input_token_index_, dict)
        self.assertTrue(hasattr(res, 'target_token_index_'))
        self.assertIsInstance(res.target_token_index_, dict)
        self.assertTrue(hasattr(res, 'reverse_target_char_index_'))
        self.assertIsInstance(res.reverse_target_char_index_, dict)
        self.assertTrue(hasattr(res, 'max_encoder_seq_length_'))
        self.assertIsInstance(res.max_encoder_seq_length_, int)
        self.assertGreater(res.max_encoder_seq_length_, 0)
        self.assertTrue(hasattr(res, 'max_decoder_seq_length_'))
        self.assertIsInstance(res.max_decoder_seq_length_, int)
        self.assertGreater(res.max_decoder_seq_length_, 0)
        self.assertTrue(hasattr(res, 'encoder_model_'))
        self.assertIsInstance(res.encoder_model_, Model)
        self.assertTrue(hasattr(res, 'encoder_model_'))
        self.assertIsInstance(res.decoder_model_, Model)

    def test_fit_positive03(self):
        """ Input and target texts for training are the Python lists. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM()
        res = seq2seq.fit(list(input_texts_for_training), list(target_texts_for_training))
        self.assertIsInstance(res, Seq2SeqLSTM)
        self.assertTrue(hasattr(res, 'input_token_index_'))
        self.assertIsInstance(res.input_token_index_, dict)
        self.assertTrue(hasattr(res, 'target_token_index_'))
        self.assertIsInstance(res.target_token_index_, dict)
        self.assertTrue(hasattr(res, 'reverse_target_char_index_'))
        self.assertIsInstance(res.reverse_target_char_index_, dict)
        self.assertTrue(hasattr(res, 'max_encoder_seq_length_'))
        self.assertIsInstance(res.max_encoder_seq_length_, int)
        self.assertGreater(res.max_encoder_seq_length_, 0)
        self.assertTrue(hasattr(res, 'max_decoder_seq_length_'))
        self.assertIsInstance(res.max_decoder_seq_length_, int)
        self.assertGreater(res.max_decoder_seq_length_, 0)
        self.assertTrue(hasattr(res, 'encoder_model_'))
        self.assertIsInstance(res.encoder_model_, Model)
        self.assertTrue(hasattr(res, 'encoder_model_'))
        self.assertIsInstance(res.decoder_model_, Model)

    def test_fit_positive04(self):
        """ Early stopping is not used in the training process. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None)
        res = seq2seq.fit(input_texts_for_training, target_texts_for_training)
        self.assertIsInstance(res, Seq2SeqLSTM)
        self.assertTrue(hasattr(res, 'input_token_index_'))
        self.assertIsInstance(res.input_token_index_, dict)
        self.assertTrue(hasattr(res, 'target_token_index_'))
        self.assertIsInstance(res.target_token_index_, dict)
        self.assertTrue(hasattr(res, 'reverse_target_char_index_'))
        self.assertIsInstance(res.reverse_target_char_index_, dict)
        self.assertTrue(hasattr(res, 'max_encoder_seq_length_'))
        self.assertIsInstance(res.max_encoder_seq_length_, int)
        self.assertGreater(res.max_encoder_seq_length_, 0)
        self.assertTrue(hasattr(res, 'max_decoder_seq_length_'))
        self.assertIsInstance(res.max_decoder_seq_length_, int)
        self.assertGreater(res.max_decoder_seq_length_, 0)
        self.assertTrue(hasattr(res, 'encoder_model_'))
        self.assertIsInstance(res.encoder_model_, Model)
        self.assertTrue(hasattr(res, 'encoder_model_'))
        self.assertIsInstance(res.decoder_model_, Model)

    def test_fit_positive05(self):
        """ Prepared evaluation set is used in the early stopping criterion. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None)
        res = seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                          eval_set=(input_texts_for_training[-20:], target_texts_for_training[-20:]))
        self.assertIsInstance(res, Seq2SeqLSTM)
        self.assertTrue(hasattr(res, 'input_token_index_'))
        self.assertIsInstance(res.input_token_index_, dict)
        self.assertTrue(hasattr(res, 'target_token_index_'))
        self.assertIsInstance(res.target_token_index_, dict)
        self.assertTrue(hasattr(res, 'reverse_target_char_index_'))
        self.assertIsInstance(res.reverse_target_char_index_, dict)
        self.assertTrue(hasattr(res, 'max_encoder_seq_length_'))
        self.assertIsInstance(res.max_encoder_seq_length_, int)
        self.assertGreater(res.max_encoder_seq_length_, 0)
        self.assertTrue(hasattr(res, 'max_decoder_seq_length_'))
        self.assertIsInstance(res.max_decoder_seq_length_, int)
        self.assertGreater(res.max_decoder_seq_length_, 0)
        self.assertTrue(hasattr(res, 'encoder_model_'))
        self.assertIsInstance(res.encoder_model_, Model)
        self.assertTrue(hasattr(res, 'encoder_model_'))
        self.assertIsInstance(res.decoder_model_, Model)

    def test_fit_negative01(self):
        """ Object with input texts is not one of the basic sequence types. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`{0}` is wrong type for `{1}`.'.format(type({1, 2}), u'X'))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            seq2seq.fit(set(input_texts_for_training), target_texts_for_training)

    def test_fit_negative02(self):
        """ Object with target texts is not one of the basic sequence types. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`{0}` is wrong type for `{1}`.'.format(type({1, 2}), u'y'))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training, set(target_texts_for_training))

    def test_fit_negative03(self):
        """ Number of input texts does not equal to number of target texts. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`X` does not correspond to `y`! {0} != {1}.'.format(
            len(input_texts_for_training), len(target_texts_for_training) - 1))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training, target_texts_for_training[:-1])

    def test_fit_negative04(self):
        """ Some parameter of the `Seq2SeqLSTM` object is wrong. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM(batch_size=0)
        true_err_msg = re.escape(u'`batch_size` must be a positive number! 0 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training, target_texts_for_training)

    def test_fit_negative05(self):
        """ Special evaluation set is neither list nor tuple. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None)
        true_err_msg = re.escape(u'`eval_set` must be `{0}` or `{1}`, not `{2}`!'.format(
            type((1, 2)), type([1, 2]), type({1: 'a', 2: 'b'})))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                        eval_set={'X': input_texts_for_training[-20:], 'y': target_texts_for_training[-20:]})

    def test_fit_negative06(self):
        """ Special evaluation set is not a two-element tuple. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None)
        true_err_msg = re.escape(u'`eval_set` must be a two-element sequence! 3 != 2')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                        eval_set=(input_texts_for_training[-20:], target_texts_for_training[-20:], [3, 4]))

    def test_fit_negative07(self):
        """ Object with input texts in the special evaluation set is not one of the basic sequence types. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`{0}` is wrong type for `{1}`.'.format(type({1, 2}), u'X_eval_set'))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                        eval_set=(set(input_texts_for_training[-20:]), target_texts_for_training[-20:]))

    def test_fit_negative08(self):
        """ Object with target texts in the special evaluation set is not one of the basic sequence types. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`{0}` is wrong type for `{1}`.'.format(type({1, 2}), u'y_eval_set'))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                        eval_set=(input_texts_for_training[-20:], set(target_texts_for_training[-20:])))

    def test_fit_negative09(self):
        """ Number of input texts does not equal to number of target texts in the special evaluation set. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`X_eval_set` does not correspond to `y_eval_set`! 20 != 19.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                        eval_set=(input_texts_for_training[-20:], target_texts_for_training[-19:]))

    def test_predict_positive001(self):
        """ Part of correctly predicted texts must be greater than 0.8. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.training_set_name)
        input_texts_for_testing, target_texts_for_testing = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None)
        seq2seq.fit(input_texts_for_training, target_texts_for_training,
                    eval_set=(input_texts_for_testing, target_texts_for_testing))
        predicted_texts = seq2seq.predict(input_texts_for_testing)
        self.assertIsInstance(predicted_texts, tuple)
        self.assertEqual(len(predicted_texts), len(input_texts_for_testing))
        self.assertGreater(self.estimate(predicted_texts, target_texts_for_testing), 0.8)

    def test_predict_negative001(self):
        """ Usage of the seq2seq model for prediction without training. """
        input_texts_for_testing, _ = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None)
        with self.assertRaises(NotFittedError):
            _ = seq2seq.predict(input_texts_for_testing)

    def test_predict_negative002(self):
        """ Input texts for prediction are wrong. """
        input_texts_for_testing, target_texts_for_testing = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None)
        seq2seq.fit(input_texts_for_testing, target_texts_for_testing)
        true_err_msg = re.escape(u'`{0}` is wrong type for `{1}`.'.format(type({1, 2}), u'X'))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = seq2seq.predict(set(input_texts_for_testing))

    def test_check_X_negative001(self):
        """ All texts must be a string and have a `split` method. """
        texts = ['123', 4, '567']
        true_err_msg = re.escape(u'Sample {0} of `{1}` is wrong! This sample have not the `split` method.'.format(
            1, u'X'))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Seq2SeqLSTM.check_X(texts, u'X')

    def test_check_X_negative002(self):
        """ If list of texts is specified as the NumPy array, then it must be a 1-D array. """
        texts = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        true_err_msg = re.escape(u'`X` must be a 1-D array!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Seq2SeqLSTM.check_X(texts, u'X')

    def test_serialize_untrained(self):
        seq2seq = Seq2SeqLSTM(batch_size=128, epochs=200, latent_dim=500, validation_split=0.1, decay=0.2, dropout=0.3,
                              recurrent_dropout=0.35, grad_clipping=50.0, lr=0.01, rho=0.8, epsilon=0.2,
                              lowercase=False, verbose=True)
        with open(self.model_name, 'wb') as fp:
            pickle.dump(seq2seq, fp)
        with open(self.model_name, 'rb') as fp:
            another_seq2seq = pickle.load(fp)
        self.assertIsInstance(another_seq2seq, Seq2SeqLSTM)
        self.assertTrue(hasattr(another_seq2seq, 'batch_size'))
        self.assertEqual(another_seq2seq.batch_size, 128)
        self.assertTrue(hasattr(another_seq2seq, 'epochs'))
        self.assertEqual(another_seq2seq.epochs, 200)
        self.assertTrue(hasattr(another_seq2seq, 'latent_dim'))
        self.assertEqual(another_seq2seq.latent_dim, 500)
        self.assertTrue(hasattr(another_seq2seq, 'validation_split'))
        self.assertAlmostEqual(another_seq2seq.validation_split, 0.1)
        self.assertTrue(hasattr(another_seq2seq, 'decay'))
        self.assertAlmostEqual(another_seq2seq.decay, 0.2)
        self.assertTrue(hasattr(another_seq2seq, 'dropout'))
        self.assertAlmostEqual(another_seq2seq.dropout, 0.3)
        self.assertTrue(hasattr(another_seq2seq, 'recurrent_dropout'))
        self.assertAlmostEqual(another_seq2seq.recurrent_dropout, 0.35)
        self.assertTrue(hasattr(another_seq2seq, 'grad_clipping'))
        self.assertAlmostEqual(another_seq2seq.grad_clipping, 50.0)
        self.assertTrue(hasattr(another_seq2seq, 'lr'))
        self.assertAlmostEqual(another_seq2seq.lr, 0.01)
        self.assertTrue(hasattr(another_seq2seq, 'rho'))
        self.assertAlmostEqual(another_seq2seq.rho, 0.8)
        self.assertTrue(hasattr(another_seq2seq, 'lowercase'))
        self.assertFalse(another_seq2seq.lowercase)
        self.assertTrue(hasattr(another_seq2seq, 'verbose'))
        self.assertTrue(another_seq2seq.verbose)

    def test_serialize_trained(self):
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.training_set_name)
        input_texts_for_testing, target_texts_for_testing = self.load_text_pairs(self.testing_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None)
        seq2seq.fit(input_texts_for_training, target_texts_for_training,
                    eval_set=(input_texts_for_testing, target_texts_for_testing))
        predicted_texts_1 = seq2seq.predict(input_texts_for_testing)
        with open(self.model_name, 'wb') as fp:
            pickle.dump(seq2seq, fp)
        del seq2seq
        with open(self.model_name, 'rb') as fp:
            another_seq2seq = pickle.load(fp)
        predicted_texts_2 = another_seq2seq.predict(input_texts_for_testing)
        self.assertEqual(predicted_texts_1, predicted_texts_2)

    @staticmethod
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
                    input_texts.append(TestSeq2SeqLSTM.tokenize_text(new_input_text))
                    target_texts.append(TestSeq2SeqLSTM.tokenize_text(new_target_text))
                cur_line = fp.readline()
                line_idx += 1
        return tuple(input_texts), tuple(target_texts)

    @staticmethod
    def tokenize_text(src):
        tokens = list()
        for cur in src.split():
            tokens += list(cur)
            tokens.append('<space>')
        return tokens[:-1]

    @staticmethod
    def estimate(predicted_texts, true_texts):
        n_err = 0
        n_total = len(predicted_texts)
        for ind in range(n_total):
            if predicted_texts[ind] != true_texts[ind]:
                n_err += 1
        return (1.0 - (n_err / float(n_total)))
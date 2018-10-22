# -*- coding: utf-8 -*-

import codecs
from difflib import SequenceMatcher
import os
import pickle
import random
import re
import sys
import unittest

from keras.models import Model
import numpy as np
from sklearn.utils.validation import NotFittedError

try:
    from seq2seq_lstm import Seq2SeqLSTM
    from seq2seq_lstm.seq2seq_lstm import TextPairSequence
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from seq2seq_lstm import Seq2SeqLSTM
    from seq2seq_lstm.seq2seq_lstm import TextPairSequence


class TestSeq2SeqLSTM(unittest.TestCase):
    def setUp(self):
        self.data_set_name = os.path.join(os.path.dirname(__file__), '..', 'data', 'eng_rus_for_testing.txt')
        self.model_name = os.path.join(os.path.dirname(__file__), '..', 'data', 'seq2seq_lstm.pkl')

    def tearDown(self):
        if os.path.isfile(self.model_name):
            os.remove(self.model_name)

    def test_creation(self):
        seq2seq = Seq2SeqLSTM(batch_size=256, char_ngram_size=2, embedding_size=30, epochs=200, latent_dim=500,
                              validation_split=0.1, use_conv_layer=True, n_filters=32, kernel_size=5, dropout=0.4,
                              recurrent_dropout=0.1, grad_clipping=50.0, lr=0.01, rho=0.8, epsilon=0.2, lowercase=False,
                              verbose=True)
        self.assertIsInstance(seq2seq, Seq2SeqLSTM)
        self.assertTrue(hasattr(seq2seq, 'batch_size'))
        self.assertEqual(seq2seq.batch_size, 256)
        self.assertTrue(hasattr(seq2seq, 'char_ngram_size'))
        self.assertEqual(seq2seq.char_ngram_size, 2)
        self.assertTrue(hasattr(seq2seq, 'embedding_size'))
        self.assertEqual(seq2seq.embedding_size, 30)
        self.assertTrue(hasattr(seq2seq, 'epochs'))
        self.assertEqual(seq2seq.epochs, 200)
        self.assertTrue(hasattr(seq2seq, 'latent_dim'))
        self.assertEqual(seq2seq.latent_dim, 500)
        self.assertTrue(hasattr(seq2seq, 'n_filters'))
        self.assertEqual(seq2seq.n_filters, 32)
        self.assertTrue(hasattr(seq2seq, 'kernel_size'))
        self.assertEqual(seq2seq.kernel_size, 5)
        self.assertTrue(hasattr(seq2seq, 'validation_split'))
        self.assertAlmostEqual(seq2seq.validation_split, 0.1)
        self.assertTrue(hasattr(seq2seq, 'dropout'))
        self.assertAlmostEqual(seq2seq.dropout, 0.4)
        self.assertTrue(hasattr(seq2seq, 'recurrent_dropout'))
        self.assertAlmostEqual(seq2seq.recurrent_dropout, 0.1)
        self.assertTrue(hasattr(seq2seq, 'grad_clipping'))
        self.assertAlmostEqual(seq2seq.grad_clipping, 50.0)
        self.assertTrue(hasattr(seq2seq, 'lr'))
        self.assertAlmostEqual(seq2seq.lr, 0.01)
        self.assertTrue(hasattr(seq2seq, 'rho'))
        self.assertAlmostEqual(seq2seq.rho, 0.8)
        self.assertTrue(hasattr(seq2seq, 'lowercase'))
        self.assertFalse(seq2seq.lowercase)
        self.assertTrue(hasattr(seq2seq, 'use_conv_layer'))
        self.assertTrue(seq2seq.use_conv_layer)
        self.assertTrue(hasattr(seq2seq, 'verbose'))
        self.assertTrue(seq2seq.verbose)

    def test_fit_positive01(self):
        """ Input and target texts for training are the Python tuples. The Conv1D layer is not used. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(use_conv_layer=False, verbose=True, lr=1e-2, epochs=10)
        res = seq2seq.fit(tuple(input_texts_for_training), tuple(target_texts_for_training))
        self.assertIsInstance(res, Seq2SeqLSTM)
        self.assertTrue(hasattr(res, 'input_embeddings_matrix_'))
        self.assertIsNone(res.input_embeddings_matrix_)
        self.assertTrue(hasattr(res, 'output_embeddings_matrix_'))
        self.assertIsNone(res.output_embeddings_matrix_)
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
        self.assertTrue(hasattr(res, 'decoder_model_'))
        self.assertIsInstance(res.decoder_model_, Model)

    def test_fit_positive02(self):
        """ The Conv1D layer is used. """
        char_ngram_size = 2
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(char_ngram_size=char_ngram_size, embedding_size=20, use_conv_layer=True, verbose=True,
                              lr=1e-2, batch_size=16, epochs=10)
        res = seq2seq.fit(tuple(input_texts_for_training), tuple(target_texts_for_training))
        self.assertIsInstance(res, Seq2SeqLSTM)
        self.assertTrue(hasattr(res, 'input_embeddings_matrix_'))
        self.assertIsInstance(res.input_embeddings_matrix_, np.ndarray)
        self.assertEqual(res.input_embeddings_matrix_.ndim, 2)
        self.assertTrue(hasattr(res, 'output_embeddings_matrix_'))
        self.assertIsInstance(res.output_embeddings_matrix_, np.ndarray)
        self.assertEqual(res.output_embeddings_matrix_.ndim, 2)
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
        self.assertTrue(hasattr(res, 'decoder_model_'))
        self.assertIsInstance(res.decoder_model_, Model)
        for cur in (set(res.input_token_index_.keys()) | set(res.target_token_index_.keys()) -
                    {(Seq2SeqLSTM.START_CHAR,), (Seq2SeqLSTM.END_CHAR,)}):
            self.assertIsInstance(cur, tuple)
            self.assertEqual(len(cur), char_ngram_size, msg=cur)

    def test_fit_positive03(self):
        """ Input and target texts for training are the 1-D numpy arrays. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(lr=1e-2, epochs=10)
        res = seq2seq.fit(np.array(input_texts_for_training), np.array(target_texts_for_training))
        self.assertIsInstance(res, Seq2SeqLSTM)
        self.assertTrue(hasattr(res, 'input_embeddings_matrix_'))
        self.assertIsNone(res.input_embeddings_matrix_)
        self.assertTrue(hasattr(res, 'output_embeddings_matrix_'))
        self.assertIsNone(res.output_embeddings_matrix_)
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
        self.assertTrue(hasattr(res, 'decoder_model_'))
        self.assertIsInstance(res.decoder_model_, Model)

    def test_fit_positive04(self):
        """ Input and target texts for training are the Python lists. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(lr=1e-2, epochs=10)
        res = seq2seq.fit(input_texts_for_training, target_texts_for_training)
        self.assertIsInstance(res, Seq2SeqLSTM)
        self.assertTrue(hasattr(res, 'input_embeddings_matrix_'))
        self.assertIsNone(res.input_embeddings_matrix_)
        self.assertTrue(hasattr(res, 'output_embeddings_matrix_'))
        self.assertIsNone(res.output_embeddings_matrix_)
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
        self.assertTrue(hasattr(res, 'decoder_model_'))
        self.assertIsInstance(res.decoder_model_, Model)

    def test_fit_positive05(self):
        """ Early stopping is not used in the training process. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None, lr=1e-2, epochs=10)
        res = seq2seq.fit(input_texts_for_training, target_texts_for_training)
        self.assertIsInstance(res, Seq2SeqLSTM)
        self.assertTrue(hasattr(res, 'input_embeddings_matrix_'))
        self.assertIsNone(res.input_embeddings_matrix_)
        self.assertTrue(hasattr(res, 'output_embeddings_matrix_'))
        self.assertIsNone(res.output_embeddings_matrix_)
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
        self.assertTrue(hasattr(res, 'decoder_model_'))
        self.assertIsInstance(res.decoder_model_, Model)

    def test_fit_positive06(self):
        """ Prepared evaluation set is used in the early stopping criterion. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None, lr=1e-2, epochs=10)
        res = seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                          eval_set=(input_texts_for_training[-20:], target_texts_for_training[-20:]))
        self.assertIsInstance(res, Seq2SeqLSTM)
        self.assertTrue(hasattr(res, 'input_embeddings_matrix_'))
        self.assertIsNone(res.input_embeddings_matrix_)
        self.assertTrue(hasattr(res, 'output_embeddings_matrix_'))
        self.assertIsNone(res.output_embeddings_matrix_)
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
        self.assertTrue(hasattr(res, 'decoder_model_'))
        self.assertIsInstance(res.decoder_model_, Model)

    def test_fit_negative01(self):
        """ Object with input texts is not one of the basic sequence types. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`{0}` is wrong type for `{1}`.'.format(type({1, 2}), u'X'))
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            seq2seq.fit(set(input_texts_for_training), target_texts_for_training)

    def test_fit_negative02(self):
        """ Object with target texts is not one of the basic sequence types. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`{0}` is wrong type for `{1}`.'.format(type({1, 2}), u'y'))
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training, set(target_texts_for_training))

    def test_fit_negative03(self):
        """ Number of input texts does not equal to number of target texts. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`X` does not correspond to `y`! {0} != {1}.'.format(
            len(input_texts_for_training), len(target_texts_for_training) - 1))
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training, target_texts_for_training[:-1])

    def test_fit_negative04(self):
        """ Some parameter of the `Seq2SeqLSTM` object is wrong. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(batch_size=0)
        true_err_msg = re.escape(u'`batch_size` must be a positive number! 0 is not positive.')
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training, target_texts_for_training)

    def test_fit_negative05(self):
        """ Special evaluation set is neither list nor tuple. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None)
        true_err_msg = re.escape(u'`eval_set` must be `{0}` or `{1}`, not `{2}`!'.format(
            type((1, 2)), type([1, 2]), type({1: 'a', 2: 'b'})))
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                        eval_set={'X': input_texts_for_training[-20:], 'y': target_texts_for_training[-20:]})

    def test_fit_negative06(self):
        """ Special evaluation set is not a two-element tuple. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None)
        true_err_msg = re.escape(u'`eval_set` must be a two-element sequence! 3 != 2')
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                        eval_set=(input_texts_for_training[-20:], target_texts_for_training[-20:], [3, 4]))

    def test_fit_negative07(self):
        """ Object with input texts in the special evaluation set is not one of the basic sequence types. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`{0}` is wrong type for `{1}`.'.format(type({1, 2}), u'X_eval_set'))
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                        eval_set=(set(input_texts_for_training[-20:]), target_texts_for_training[-20:]))

    def test_fit_negative08(self):
        """ Object with target texts in the special evaluation set is not one of the basic sequence types. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`{0}` is wrong type for `{1}`.'.format(type({1, 2}), u'y_eval_set'))
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                        eval_set=(input_texts_for_training[-20:], set(target_texts_for_training[-20:])))

    def test_fit_negative09(self):
        """ Number of input texts does not equal to number of target texts in the special evaluation set. """
        input_texts_for_training, target_texts_for_training = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM()
        true_err_msg = re.escape(u'`X_eval_set` does not correspond to `y_eval_set`! 20 != 19.')
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            seq2seq.fit(input_texts_for_training[:-20], target_texts_for_training[:-20],
                        eval_set=(input_texts_for_training[-20:], target_texts_for_training[-19:]))

    def test_predict_positive001(self):
        """ Part of correctly predicted texts must be greater than 0.5. """
        input_texts, target_texts = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None, epochs=100, lr=1e-2, verbose=True, lowercase=False)
        predicted_texts = seq2seq.fit_predict(input_texts, target_texts)
        self.assertIsInstance(predicted_texts, list)
        self.assertEqual(len(predicted_texts), len(input_texts))
        self.assertGreater(self.estimate(predicted_texts, target_texts), 0.5)

    def test_predict_positive002(self):
        """ Part of correctly predicted texts must be greater than 0.5. """
        input_texts, target_texts = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(char_ngram_size=2, embedding_size=30, validation_split=None, epochs=50, lr=1e-2,
                              verbose=True, lowercase=True, batch_size=64)
        predicted_texts = seq2seq.fit_predict(input_texts, target_texts)
        self.assertIsInstance(predicted_texts, list)
        self.assertEqual(len(predicted_texts), len(input_texts))
        self.assertGreater(self.estimate(predicted_texts, target_texts), 0.5)

    def test_predict_negative001(self):
        """ Usage of the seq2seq model for prediction without training. """
        input_texts_for_testing, _ = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None, epochs=5)
        with self.assertRaises(NotFittedError):
            _ = seq2seq.predict(input_texts_for_testing)

    def test_predict_negative002(self):
        """ Input texts for prediction are wrong. """
        input_texts_for_testing, target_texts_for_testing = self.load_text_pairs(self.data_set_name)
        seq2seq = Seq2SeqLSTM(validation_split=None, epochs=5)
        seq2seq.fit(input_texts_for_testing, target_texts_for_testing)
        true_err_msg = re.escape(u'`{0}` is wrong type for `{1}`.'.format(type({1, 2}), u'X'))
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            _ = seq2seq.predict(set(input_texts_for_testing))

    def test_check_X_negative001(self):
        """ All texts must be a string and have a `split` method. """
        texts = [u'123', 4, u'567']
        true_err_msg = re.escape(u'Sample {0} of `{1}` is wrong! This sample have not the `split` method.'.format(
            1, u'X'))
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            Seq2SeqLSTM.check_X(texts, u'X')

    def test_check_X_negative002(self):
        """ If list of texts is specified as the NumPy array, then it must be a 1-D array. """
        texts = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
        true_err_msg = re.escape(u'`X` must be a 1-D array!')
        try:
            checking_method = self.assertRaisesRegex
        except:
            checking_method = self.assertRaisesRegexp
        with checking_method(ValueError, true_err_msg):
            Seq2SeqLSTM.check_X(texts, u'X')

    def test_serialize_untrained(self):
        seq2seq = Seq2SeqLSTM(batch_size=256, char_ngram_size=3, epochs=200, latent_dim=500, validation_split=0.1,
                              dropout=0.7, recurrent_dropout=0.2, grad_clipping=50.0, lr=0.01, rho=0.8, epsilon=0.2,
                              lowercase=False, verbose=True)
        with open(self.model_name, 'wb') as fp:
            pickle.dump(seq2seq, fp)
        with open(self.model_name, 'rb') as fp:
            another_seq2seq = pickle.load(fp)
        self.assertIsInstance(another_seq2seq, Seq2SeqLSTM)
        self.assertTrue(hasattr(another_seq2seq, 'batch_size'))
        self.assertEqual(another_seq2seq.batch_size, 256)
        self.assertTrue(hasattr(another_seq2seq, 'char_ngram_size'))
        self.assertEqual(another_seq2seq.char_ngram_size, 3)
        self.assertTrue(hasattr(another_seq2seq, 'embedding_size'))
        self.assertIsNone(another_seq2seq.embedding_size)
        self.assertTrue(hasattr(another_seq2seq, 'epochs'))
        self.assertEqual(another_seq2seq.epochs, 200)
        self.assertTrue(hasattr(another_seq2seq, 'latent_dim'))
        self.assertEqual(another_seq2seq.latent_dim, 500)
        self.assertTrue(hasattr(another_seq2seq, 'validation_split'))
        self.assertAlmostEqual(another_seq2seq.validation_split, 0.1)
        self.assertTrue(hasattr(another_seq2seq, 'dropout'))
        self.assertAlmostEqual(another_seq2seq.dropout, 0.7)
        self.assertTrue(hasattr(another_seq2seq, 'recurrent_dropout'))
        self.assertAlmostEqual(another_seq2seq.recurrent_dropout, 0.2)
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
        input_texts, target_texts = self.load_text_pairs(self.data_set_name)
        indices = list(range(len(input_texts)))
        random.shuffle(indices)
        n = int(round(0.2 * len(indices)))
        input_texts_for_training = []
        target_texts_for_training = []
        for ind in indices[:-n]:
            input_texts_for_training.append(input_texts[ind])
            target_texts_for_training.append(target_texts[ind])
        input_texts_for_testing = []
        target_texts_for_testing = []
        for ind in indices[-n:]:
            input_texts_for_testing.append(input_texts[ind])
            target_texts_for_testing.append(target_texts[ind])
        seq2seq = Seq2SeqLSTM(validation_split=None, embedding_size=5, epochs=10, lr=1e-3)
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

    def test_tokenize_text_positive01(self):
        """ Tokenization with saving of the characters register. """
        src = u'a\t B  c Мама мыла \n\r раму 1\n'
        dst_true = [u'a', u'B', u'c', u'Мама', u'мыла', u'раму', u'1']
        dst_predicted = Seq2SeqLSTM.tokenize_text(src, lowercase=False)
        self.assertEqual(dst_predicted, dst_true)

    def test_tokenize_text_positive02(self):
        """ Tokenization with bringing the resulting tokens to lowercase. """
        src = u'a\t B  c Мама мыла \n\r раму 1\n'
        dst_true = [u'a', u'b', u'c', u'мама', u'мыла', u'раму', u'1']
        dst_predicted = Seq2SeqLSTM.tokenize_text(src, lowercase=True)
        self.assertEqual(dst_predicted, dst_true)

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
        n = len(input_texts)
        if n > 500:
            indices = [idx for idx in range(n)]
            random.shuffle(indices)
            indices = indices[:500]
            input_texts = [input_texts[idx] for idx in indices]
            target_texts = [target_texts[idx] for idx in indices]
        return input_texts, target_texts

    @staticmethod
    def tokenize_text(src):
        tokens = list()
        for cur in src.split():
            tokens += list(cur)
            tokens.append(u'<space>')
        return u' '.join(tokens[:-1])

    @staticmethod
    def detokenize_text(src):
        new_text = u''
        for cur_token in src.split():
            if cur_token == u'<space>':
                new_text += u' '
            else:
                new_text += cur_token
        return new_text.strip()

    @staticmethod
    def estimate(predicted_texts, true_texts):
        n_total = len(predicted_texts)
        similarity = 0.0
        for i in range(n_total):
            cur_predicted = TestSeq2SeqLSTM.detokenize_text(predicted_texts[i]).lower()
            cur_true = TestSeq2SeqLSTM.detokenize_text(true_texts[i]).lower()
            similarity += SequenceMatcher(a=cur_predicted, b=cur_true).ratio()
        return similarity / float(n_total)


class TestTextPairSequence(unittest.TestCase):
    def characters_to_ngrams_1(self):
        input_text = u'a b c 1 2 d'
        true_ngrams = [(Seq2SeqLSTM.START_CHAR, u'a', u'b'), (u'a', u'b', u'c'), (u'b', u'c', u'1'), (u'c', u'1', u'2'),
                       (u'1', u'2', u'd'), (u'2', u'd', Seq2SeqLSTM.END_CHAR)]
        self.assertEqual(true_ngrams, Seq2SeqLSTM.characters_to_ngrams(input_text.split(), 3, False, False))

    def characters_to_ngrams_2(self):
        input_text = u'a b c 1 2 d'
        true_ngrams = [(Seq2SeqLSTM.START_CHAR,), (Seq2SeqLSTM.START_CHAR, u'a', u'b'), (u'a', u'b', u'c'),
                       (u'b', u'c', u'1'), (u'c', u'1', u'2'), (u'1', u'2', u'd'), (u'2', u'd', Seq2SeqLSTM.END_CHAR)]
        self.assertEqual(true_ngrams, Seq2SeqLSTM.characters_to_ngrams(input_text.split(), 3, True, False))

    def characters_to_ngrams_3(self):
        input_text = u'a b c 1 2 d'
        true_ngrams = [(Seq2SeqLSTM.START_CHAR, u'a', u'b'), (u'a', u'b', u'c'), (u'b', u'c', u'1'), (u'c', u'1', u'2'),
                       (u'1', u'2', u'd'), (u'2', u'd', Seq2SeqLSTM.END_CHAR), (Seq2SeqLSTM.END_CHAR,)]
        self.assertEqual(true_ngrams, Seq2SeqLSTM.characters_to_ngrams(input_text.split(), 3, False, True))

    def characters_to_ngrams_4(self):
        input_text = u'a b c 1 2 d'
        true_ngrams = [(Seq2SeqLSTM.START_CHAR,), (Seq2SeqLSTM.START_CHAR, u'a', u'b'), (u'a', u'b', u'c'),
                       (u'b', u'c', u'1'), (u'c', u'1', u'2'), (u'1', u'2', u'd'), (u'2', u'd', Seq2SeqLSTM.END_CHAR),
                       (Seq2SeqLSTM.END_CHAR,)]
        self.assertEqual(true_ngrams, Seq2SeqLSTM.characters_to_ngrams(input_text.split(), 3, True, True))

    def test_generate_data_for_training_1(self):
        input_texts = [
            u'a b c',
            u'a c',
            u'0 1 b',
            u'b a',
            u'b c'
        ]
        target_texts = [
            u'а б а 2',
            u'2 3',
            u'а б а',
            u'б а',
            u'б 3'
        ]
        char_ngram_size = 1
        batch_size = 2
        max_encoder_seq_length = 3
        max_decoder_seq_length = 6
        input_token_index = {(u'0',): 0, (u'1',): 1, (u'a',): 2, (u'b',): 3, (u'c',): 4}
        target_token_index = {(Seq2SeqLSTM.START_CHAR,): 0, (Seq2SeqLSTM.END_CHAR,): 1, (u'2',): 2, (u'3',): 3,
                              (u'а',): 4, (u'б',): 5}
        true_batches = [
            (
                [
                    np.array([
                        [
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]
                        ],
                        [
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]
                        ]
                    ]),
                    np.array([
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                        ],
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ])
                ],
                np.array([
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ]
                ])
            ),
            (
                [
                    np.array([
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0]
                        ],
                        [
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]
                        ]
                    ]),
                    np.array([
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        ],
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        ]
                    ])
                ],
                np.array([
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ]
                ])
            ),
            (
                [
                    np.array([
                        [
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]
                        ],
                        [
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0]
                        ]
                    ]),
                    np.array([
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        ],
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                        ]
                    ])
                ],
                np.array([
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ]
                ])
            )
        ]
        training_set_generator = TextPairSequence(
            input_texts=input_texts, target_texts=target_texts, batch_size=batch_size, char_ngram_size=char_ngram_size,
            max_encoder_seq_length=max_encoder_seq_length, max_decoder_seq_length=max_decoder_seq_length,
            input_token_index=input_token_index, target_token_index=target_token_index, lowercase=False,
            use_embeddings=False
        )
        self.assertIsInstance(training_set_generator, TextPairSequence)
        self.assertTrue(hasattr(training_set_generator, 'input_texts'))
        self.assertTrue(hasattr(training_set_generator, 'target_texts'))
        self.assertTrue(hasattr(training_set_generator, 'batch_size'))
        self.assertTrue(hasattr(training_set_generator, 'char_ngram_size'))
        self.assertTrue(hasattr(training_set_generator, 'max_encoder_seq_length'))
        self.assertTrue(hasattr(training_set_generator, 'max_decoder_seq_length'))
        self.assertTrue(hasattr(training_set_generator, 'input_token_index'))
        self.assertTrue(hasattr(training_set_generator, 'target_token_index'))
        self.assertTrue(hasattr(training_set_generator, 'lowercase'))
        self.assertTrue(hasattr(training_set_generator, 'n_text_pairs'))
        self.assertTrue(hasattr(training_set_generator, 'n_batches'))
        self.assertIs(training_set_generator.input_texts, input_texts)
        self.assertIs(training_set_generator.target_texts, target_texts)
        self.assertEqual(training_set_generator.batch_size, batch_size)
        self.assertEqual(training_set_generator.char_ngram_size, char_ngram_size)
        self.assertEqual(training_set_generator.max_encoder_seq_length, max_encoder_seq_length)
        self.assertEqual(training_set_generator.max_decoder_seq_length, max_decoder_seq_length)
        self.assertIs(training_set_generator.input_token_index, input_token_index)
        self.assertIs(training_set_generator.target_token_index, target_token_index)
        self.assertFalse(training_set_generator.lowercase)
        self.assertIsInstance(training_set_generator.n_text_pairs, int)
        self.assertEqual(training_set_generator.n_text_pairs, len(input_texts))
        self.assertIsInstance(training_set_generator.n_batches, int)
        self.assertEqual(training_set_generator.n_batches, len(true_batches))
        for batch_ind in range(len(true_batches)):
            predicted_batch = training_set_generator[batch_ind]
            self.assertIsInstance(predicted_batch, tuple, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertEqual(len(predicted_batch), 2, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertIsInstance(predicted_batch[0], list, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertIsInstance(predicted_batch[1], np.ndarray, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertEqual(len(predicted_batch[0]), 2, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertIsInstance(predicted_batch[0][0], np.ndarray, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertIsInstance(predicted_batch[0][1], np.ndarray, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertTrue(np.array_equal(predicted_batch[0][0], true_batches[batch_ind][0][0]),
                            msg=u'batch_ind={0}, encoder_input_data'.format(batch_ind))
            self.assertTrue(np.array_equal(predicted_batch[0][1], true_batches[batch_ind][0][1]),
                            msg=u'batch_ind={0}, decoder_input_data'.format(batch_ind))
            self.assertTrue(np.array_equal(predicted_batch[1], true_batches[batch_ind][1]),
                            msg=u'batch_ind={0}, decoder_target_data'.format(batch_ind))

    def test_generate_data_for_training_2(self):
        input_texts = [
            u'a b c',
            u'a c',
            u'0 1 b',
            u'b a',
            u'b c'
        ]
        target_texts = [
            u'а б а 2',
            u'2 3',
            u'а б а',
            u'б а',
            u'б 3'
        ]
        char_ngram_size = 1
        batch_size = 2
        max_encoder_seq_length = 3
        max_decoder_seq_length = 6
        input_token_index = {(u'0',): 0, (u'1',): 1, (u'a',): 2, (u'b',): 3, (u'c',): 4}
        target_token_index = {(Seq2SeqLSTM.START_CHAR,): 0, (Seq2SeqLSTM.END_CHAR,): 1, (u'2',): 2, (u'3',): 3,
                              (u'а',): 4, (u'б',): 5}
        true_batches = [
            (
                [
                    np.array([
                        [3, 4, 5],
                        [3, 5, 0]
                    ]),
                    np.array([
                        [1, 5, 6, 5, 3, 2],
                        [1, 3, 4, 2, 0, 0]
                    ])
                ],
                np.array([
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    [
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ]
                ])
            ),
            (
                [
                    np.array([
                        [1, 2, 4],
                        [4, 3, 0]
                    ]),
                    np.array([
                        [1, 5, 6, 5, 2, 0],
                        [1, 6, 5, 2, 0, 0]
                    ])
                ],
                np.array([
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ]
                ])
            ),
            (
                [
                    np.array([
                        [4, 5, 0],
                        [3, 4, 5]
                    ]),
                    np.array([
                        [1, 6, 4, 2, 0, 0],
                        [1, 5, 6, 5, 3, 2]
                    ])
                ],
                np.array([
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    [
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ]
                ])
            )
        ]
        training_set_generator = TextPairSequence(
            input_texts=input_texts, target_texts=target_texts, batch_size=batch_size, char_ngram_size=char_ngram_size,
            max_encoder_seq_length=max_encoder_seq_length, max_decoder_seq_length=max_decoder_seq_length,
            input_token_index=input_token_index, target_token_index=target_token_index, lowercase=False,
            use_embeddings=True
        )
        self.assertIsInstance(training_set_generator, TextPairSequence)
        self.assertTrue(hasattr(training_set_generator, 'input_texts'))
        self.assertTrue(hasattr(training_set_generator, 'target_texts'))
        self.assertTrue(hasattr(training_set_generator, 'batch_size'))
        self.assertTrue(hasattr(training_set_generator, 'char_ngram_size'))
        self.assertTrue(hasattr(training_set_generator, 'max_encoder_seq_length'))
        self.assertTrue(hasattr(training_set_generator, 'max_decoder_seq_length'))
        self.assertTrue(hasattr(training_set_generator, 'input_token_index'))
        self.assertTrue(hasattr(training_set_generator, 'target_token_index'))
        self.assertTrue(hasattr(training_set_generator, 'lowercase'))
        self.assertTrue(hasattr(training_set_generator, 'n_text_pairs'))
        self.assertTrue(hasattr(training_set_generator, 'n_batches'))
        self.assertIs(training_set_generator.input_texts, input_texts)
        self.assertIs(training_set_generator.target_texts, target_texts)
        self.assertEqual(training_set_generator.batch_size, batch_size)
        self.assertEqual(training_set_generator.char_ngram_size, char_ngram_size)
        self.assertEqual(training_set_generator.max_encoder_seq_length, max_encoder_seq_length)
        self.assertEqual(training_set_generator.max_decoder_seq_length, max_decoder_seq_length)
        self.assertIs(training_set_generator.input_token_index, input_token_index)
        self.assertIs(training_set_generator.target_token_index, target_token_index)
        self.assertFalse(training_set_generator.lowercase)
        self.assertIsInstance(training_set_generator.n_text_pairs, int)
        self.assertEqual(training_set_generator.n_text_pairs, len(input_texts))
        self.assertIsInstance(training_set_generator.n_batches, int)
        self.assertEqual(training_set_generator.n_batches, len(true_batches))
        for batch_ind in range(len(true_batches)):
            predicted_batch = training_set_generator[batch_ind]
            self.assertIsInstance(predicted_batch, tuple, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertEqual(len(predicted_batch), 2, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertIsInstance(predicted_batch[0], list, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertIsInstance(predicted_batch[1], np.ndarray, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertEqual(len(predicted_batch[0]), 2, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertIsInstance(predicted_batch[0][0], np.ndarray, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertIsInstance(predicted_batch[0][1], np.ndarray, msg=u'batch_ind={0}'.format(batch_ind))
            self.assertTrue(np.array_equal(predicted_batch[0][0], true_batches[batch_ind][0][0]),
                            msg=u'batch_ind={0}, encoder_input_data'.format(batch_ind))
            self.assertTrue(np.array_equal(predicted_batch[0][1], true_batches[batch_ind][0][1]),
                            msg=u'batch_ind={0}, decoder_input_data'.format(batch_ind))
            self.assertTrue(np.array_equal(predicted_batch[1], true_batches[batch_ind][1]),
                            msg=u'batch_ind={0}, decoder_target_data'.format(batch_ind))


if __name__ == '__main__':
    unittest.main(verbosity=2)

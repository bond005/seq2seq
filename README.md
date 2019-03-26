# seq2seq-lstm

The Seq2Seq-LSTM is a sequence-to-sequence classifier with the sklearn-like interface, and it uses the Keras package for neural modeling.

Developing of this module was inspired by this tutorial:

_Francois Chollet_, **A ten-minute introduction to sequence-to-sequence learning in Keras**, https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

The goal of this project is creating a simple Python package with the sklearn-like interface for solution of different seq2seq tasks:
machine translation, question answering, decoding phonemes sequence into the word sequence, etc.

## Getting Started

### Installing

To install this project on your local machine, you should run the following commands in Terminal:

```
git clone https://github.com/bond005/seq2seq.git
cd seq2seq
sudo python setup.py install
```

You can also run the tests

```
python setup.py test
```

But I recommend you to use pip and install this package from PyPi:

```
pip install seq2seq-lstm
```

or

```
sudo pip install seq2seq-lstm
```

### Usage

After installing the Seq2Seq-LSTM can be used as Python package in your projects. For example:

```
from seq2seq_lstm import Seq2SeqLSTM  # import the Seq2Seq-LSTM package
seq2seq = Seq2SeqLSTM()  # create new sequence-to-sequence transformer
```

To see the work of the Seq2Seq-LSTM on a large dataset, you can run a demo

```
python demo/seq2seq_lstm_demo.py
```

or

```
python demo/seq2seq_lstm_demo.py some_file.pkl
```

In this demo, the Seq2Seq-LSTM learns to translate the sentences from English into Russian. If you specify the neural model file (for example, aforementioned `some_file.pkl`), then the learned neural model will be saved into this file for its loading instead of re-fitting at the next running.

The Russian-English sentence pairs from the Tatoeba Project have been used as data for unit tests and demo script (see http://www.manythings.org/anki/).


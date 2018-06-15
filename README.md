# seq2seq-lstm

Seq2Seq-LSTM is a sequence-to-sequence classifier which has the sklearn-like interface and uses the Keras for neural modeling.

Developing of this module was inspired by this tutorial:

_Francois Chollet_, **A ten-minute introduction to sequence-to-sequence learning in Keras**, https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

Project goal is creating a simple Python package with the sklearn-like interface for solution of different seq2seq tasks:
machine translation, question answering, decoding phonemes sequence into the word sequence, etc.

## Getting Started

### Prerequisites

You should have python installed on your machine (we recommend Anaconda package) and modules listed in requirements.txt. If you do not have them, run in Terminal

```
pip install -r requirements.txt
```

### Installing and Usage

To install this project on your local machine, you should run the following commands in Terminal:

```
cd YOUR_FOLDER
git clone https://github.com/bond005/seq2seq.git
cd seq2seq
sudo python setup.py
```

You can also run the tests

```
python setup.py test
```

To see the work of the Seq2Seq-LSTM on a large dataset, you can run a demo

```
python demo/seq2seq_lstm_demo.py
```

or

```
python demo/seq2seq_lstm_demo.py some_file.pkl
```

In this demo, the Seq2Seq-LSTM learns to translate the sentences from English into Russian. If you specify the neural model file (for example, aforementioned `some_file.pkl`), then the fitted neural model will be saved into this file, and at the next startups it will be loaded from the file instead of re-learning.

The Russian-English sentence pairs from the Tatoeba Project have been used as data for unit tests and demo script (see http://www.manythings.org/anki/).


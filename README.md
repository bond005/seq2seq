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
demo/seq2seq_lstm_demo
```

or

```
python demo/seq2seq_lstm_demo.py
```

In this demo, the Seq2Seq-LSTM learns to translate the sentences from English into Russian.

The Russian-English sentence pairs from the Tatoeba Project have been used as data for unit tests (see http://www.manythings.org/anki/).


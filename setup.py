# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import seq2seq_lstm


long_description = '''
seq2seq-lstm
============

The Seq2Seq-LSTM is a sequence-to-sequence classifier with the
sklearn-like interface, and it uses the Keras package for neural
modeling.

Developing of this module was inspired by Francois Chollet's tutorial
`A ten-minute introduction to sequence-to-sequence learning in Keras
<https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html>`_

The goal of this project is creating a simple Python package with the
sklearn-like interface for solution of different seq2seq tasks: machine
translation, question answering, decoding phonemes sequence into the
word sequence, etc.

Getting Started
---------------

Installing
~~~~~~~~~~

To install this project on your local machine, you should run the
following commands in Terminal:

.. code::

    git clone https://github.com/bond005/seq2seq.git
    cd seq2seq
    sudo python setup.py

You can also run the tests:

.. code::

    python setup.py test

But I recommend you to use pip and install this package from PyPi:

.. code::

    pip install seq2seq-lstm

or (using ``sudo``):

.. code::

    sudo pip install seq2seq-lstm

Usage
~~~~~

After installing the Seq2Seq-LSTM can be used as Python package in your
projects. For example:

.. code::

    from seq2seq import Seq2SeqLSTM  # import the Seq2Seq-LSTM package
    seq2seq = Seq2SeqLSTM()  # create new sequence-to-sequence transformer

To see the work of the Seq2Seq-LSTM on a large dataset, you can run a
demo

.. code::
 
    python demo/seq2seq_lstm_demo.py

or (with saving model after its training):

.. code::
 
    python demo/seq2seq_lstm_demo.py some_file.pkl

In this demo, the Seq2Seq-LSTM learns to translate the sentences from
English into Russian. If you specify the neural model file (for example,
aforementioned ``some_file.pkl``), then the learned neural model will be
saved into this file for its loading instead of re-fitting at the next
running.

The Russian-English sentence pairs from the Tatoeba Project have been
used as data for unit tests and demo script (see
`http://www.manythings.org/anki/ <http://www.manythings.org/anki/>`_).

'''

setup(
    name='seq2seq-lstm',
    version=seq2seq_lstm.__version__,
    packages=find_packages(exclude=['tests', 'demo']),
    include_package_data=True,
    description='Sequence-to-sequence classifier based on LSTM with the simple sklearn-like interface',
    long_description=long_description,
    url='https://github.com/bond005/seq2seq',
    author='Ivan Bondarenko',
    author_email='bond005@yandex.ru',
    license='Apache License Version 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords=['seq2seq', 'sequence-to-sequence', 'lstm', 'nlp', 'keras', 'scikit-learn'],
    install_requires=['h5py>=2.10.0', 'tensorflow>=2.2.1', 'numpy>=1.18.5', 'scikit-learn>=0.23.2'],
    test_suite='tests'
)

# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from codecs import open
from os.path import dirname, join

import seq2seq_lstm


setup(
    name='seq2seq-lstm',
    version=seq2seq_lstm.__version__,
    packages=find_packages(exclude=['tests', 'demo']),
    include_package_data=True,
    description='Sequence-to-sequence classifier based on LSTM with the simple sklearn-like interface',
    long_description=open(join(dirname(__file__), 'README.md'), encoding='utf-8').read(),
    long_description_content_type="text/markdown",
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
    install_requires=['h5py', 'keras', 'numpy', 'scikit-learn'],
    test_suite='tests'
)

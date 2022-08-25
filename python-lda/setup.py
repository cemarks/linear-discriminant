#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='LinearDiscriminant',
    description='Linear Discriminant Analysis',
    author = 'Christopher Marks',
    author_email = 'cemarks@alum.mit.edu',
    version='1.0',
    packages = find_packages(),
    install_requires = [
        'matplotlib',
        'scikit-learn',
        'scipy',
        'numpy'
    ]
)


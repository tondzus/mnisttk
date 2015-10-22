#!/usr/env/bin python

from setuptools import setup


setup(
    name='mnisttk',
    url='http://github.com/tondzus/mnisttk/',
    version='0.1',
    description='Library to help work with mnist hand written digit database',
    author='Tomas Stibrany',
    author_email='tms.stibrany@gmail.com',
    install_requires=['numpy'],
    py_modules=['mnisttk'],
)

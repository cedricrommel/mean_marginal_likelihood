# !/usr/bin/env python
# -*- coding:utf-8 -*-
#
# Author: C. Rommel
# Safety Line, 2018

from setuptools import setup, find_packages
import sys
import os


if '--v' in sys.argv:
    idx = sys.argv.index('--v')
    version = sys.argv[idx+1]
    sys.argv.remove('--v')
    sys.argv.remove(version)

    print('\n----------------------------------------\n')
    print('--- Building Package v{} ---\n'.format(version))
    print('----------------------------------------\n\n')

    def readme():
        with open('README.md') as f:
            return f.read()

    print('--- Build package ---\n')
    setup(
        name='accept',
        version=version,
        author='Cedric Rommel - Safety Line',
        packages=find_packages(),
        license='LICENSE.txt',
        description='Python package for statistical acceptability assessment.',
        long_description=readme(),
        classifiers=['Programming Language :: Python :: 2.7'],
        install_requires=['numpy', 'sklearn', 'pandas',
                          'matplotlib', 'seaborn', 'scipy', 'fastkde', 'densratio'],
        include_package_data=True)

    if 'clean' in sys.argv:
        s = 'accept-{}.tar.gz'.format(version)
        command = 'rm dist/'+s
        print('\n' + command)
        os.system(command)

else:
    print('Error: version missing')
    print('Usage: python setup.py [install or sdist] --v [version]')

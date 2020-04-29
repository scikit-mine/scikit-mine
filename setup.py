#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import find_packages, dist
import distutils.util
from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import os
from glob import glob
from packaging.version import parse
import skmine

import numpy

parsed_version = parse(skmine.__version__)
if parsed_version.is_postrelease:
    release = parsed_version.base_version
else:
    release = skmine.__version__


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as req_fd:
    reqs = req_fd.read().splitlines()

setup(
    author="Scikit-network team",
    author_email='bonald@enst.fr',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    description="Graph algorithms",
    entry_points={
        'console_scripts': [
            'skmine=skmine.cli:main',
        ],
    },
    install_requires=reqs,
    license="BSD license",
    long_description=readme,
    long_description_content_type='text/x-rst',
    #include_package_data=True,
    keywords='skmine',
    name='scikit-mine',
    packages=['skmine'],
    test_suite='tests',
    tests_require=['pytest'],
    url='https://github.com/scikit-mine/scikit-mine',
    version=release,
    zip_safe=False,
    #ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)
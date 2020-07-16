#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import find_packages, setup
import numpy

import skmine

DISTNAME = 'scikit-mine'
DESCRIPTION = 'Pattern mining in Python'
MAINTAINER = 'R. Adon'
MAINTAINER_EMAIL = 'remi.adon@gmail.com'
URL = 'https://github.com/scikit-mine/scikit-mine'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/scikit-mine/scikit-mine'
VERSION = skmine.__version__
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']

EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}


# try replacing with `codecs.open('README.rst', encoding='utf-8-sig') as f:` if error
with open('README.rst') as readme_file:
    LONG_DESCRIPTION = readme_file.read()

with open('requirements.txt') as req_fd:
    INSTALL_REQUIRES = req_fd.read().splitlines()


setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/x-rst',
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      include_dirs=[numpy.get_include()],
)

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from packaging.version import parse

sys.path.insert(0, os.path.abspath('../'))

import skmine

# -- Project information -----------------------------------------------------
project = 'scikit-mine'
copyright = '2022, scikit-mine team'
author = 'scikit-mine team'

# The full version
parsed_version = parse(skmine.__version__)
if parsed_version.is_postrelease:
    release = parsed_version.base_version
else:
    release = skmine.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

autodoc_default_options = {
    'members': True,
    'inherited-members': False
}

autosummary_generate = True

autodoc_member_order = 'bysource'

napoleon_use_ivar = True

numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    'modules.rst',
    'skmine*.rst',
]


source_suffix = '.rst'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# With this enabled, navigation entries are not expandable – the [+] icons next to each entry are removed.
collapse_navigation = True


# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


html_theme = 'sphinx_rtd_theme'
html_logo = 'skmine_logo.png'
html_theme_options = {'logo_only': True}
html_scaled_image_link = False

nbsphinx_execute = 'always'
nbsphinx_kernel_name = 'python3'
nbsphinx_execute = 'never'

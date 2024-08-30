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
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'microscopy_data_analysis docs'
copyright = '2024, Kernke'
author = 'Robert Kernke'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc','sphinx.ext.viewcode','sphinx.ext.todo', 'nbsphinx_link'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store','deprecated.py','under_construction.py']


# -- Settings for autoapi extension ----------------------------

# autoapi gets the docstrings for all public modules in the package
autoapi_type = 'python'
autoapi_dirs = ['../microscopy_data_analysis']
autoapi_template_dir = '_templates/autoapi'
autoapi_root = 'api'
autoapi_options = [
    'members',
    'inherited-members',
    #'undoc-members', # show objects that do not have doc strings
    #'private_members', # show private objects (_variable)
    #'show-inheritance',
    'show-module-summary',
    #'special-members', # show things like __str__
    #'imported-members', # document things imported within each module
]
autoapi_member_order = 'groupwise'  # groups into classes, functions, etc.
autoapi_python_class_content = 'class'  # include class docstring from class and/or __init__
autoapi_keep_files = False  # keep the files after generation
autoapi_add_toctree_entry = True  # need to manually add to toctree if False
autoapi_generate_api_docs = True  # will not generate new docs when False

# ignore an import warning from sphinx-autoapi due to double import of utils
suppress_warnings = ['autoapi.python_import_resolution', 'autosectionlabel']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"#"pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []#'_static']
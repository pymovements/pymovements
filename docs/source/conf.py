# Copyright (c) 2022-2025 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
import importlib.resources
import inspect
import os
import sys
from subprocess import CalledProcessError
from subprocess import run

from pybtex.plugin import register_plugin
from pybtex.style.formatting.plain import Style as PlainStyle
from pybtex.style.labels import BaseLabelStyle

# add relative source path to python path
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.dirname(os.path.abspath('src')))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath('src'))))


# -- Project information -----------------------------------------------------

project = 'pymovements'
copyright = '2022-2025 The pymovements Project Authors'
author = 'The pymovements Project Authors'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_favicon',
    'sphinx_mdinclude',
    'sphinxcontrib.datatemplates',
    'sphinxcontrib.bibtex',
    'myst_nb',  # load after `sphinx_mdinclude` to supress extension error ('.md' registration)
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
    '.md': 'markdown',
}


def config_inited_handler(app, config):
    os.makedirs(os.path.join(app.srcdir, app.config.generated_path), exist_ok=True)


def setup(app):
    app.add_config_value('REVISION', 'master', 'env')
    app.add_config_value('generated_path', '_generated', 'env')
    app.connect('config-inited', config_inited_handler)


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []
suppress_warnings = [
    'myst.header',
]


copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: '
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = '\\'
copybutton_here_doc_delimiter = 'EOT'

# -- Options for cross-references ---------------------------------------------------

# Help Napoleon resolve common short type names in docstrings
napoleon_type_aliases = {
    # Builtins / stdlib typing shorthands
    'Sequence': 'collections.abc.Sequence',
    'Iterable': 'collections.abc.Iterable',
    'Callable': 'collections.abc.Callable',

    # Datetime
    'datetime': 'datetime.datetime',

    # Filesystem paths
    'Path': 'pathlib.Path',

    # NumPy core
    'ndarray': 'numpy.ndarray',
    'np.ndarray': 'numpy.ndarray',
    'DTypeLike': 'numpy.typing.DTypeLike',

    # Pandas
    'pd.DataFrame': 'pandas.DataFrame',

    # Matplotlib common types
    'plt.Figure': 'matplotlib.figure.Figure',
    'plt.Axes': 'matplotlib.axes.Axes',
    'colors.Colormap': 'matplotlib.colors.Colormap',
    'colors.Normalize': 'matplotlib.colors.Normalize',

    # Polars (broken intersphinx)
    'pl.DataFrame': 'polars.DataFrame',
    'pl.Series': 'polars.Series',
    'pl.Expr': 'polars.Expr',
}

nitpicky = True
# Patterns for ignoring nitpicky cross-ref warnings. The regex matches the TARGET only,
# not the full warning message. Keep these as narrow as possible.
# See: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpick_ignore_regex
nitpick_ignore_regex = [
    # pathlib Path types used in type hints/docstrings which are not resolvable in our docs
    (r'py:class', r'^(?:pathlib\._local\.)?Path$'),

    # Numpy shorthand types that are usually written as np.X but not resolvable
    (r'py:class', r'^np\..*'),

    # Polars types referenced with either pl.X or fully qualified polars.*
    # Context: polars intersphinx mapping is broken, https://github.com/pola-rs/polars/issues/7027
    (r'py:(class|mod)', r'^(?:pl|polars)(?:\..*)?$'),

    # Allow explicit pandas shorthand in RST roles that napoleon cannot rewrite
    (r'py:class', r'^pd\.DataFrame$'),

    # Matplotlib pyplot short alias references like plt.X
    (r'py:(class|mod|func|meth|obj|attr)', r'^plt\..*'),

    # Our docs might reference a plain "transforms.func" symbol coming from context
    (r'py:(func|meth|mod)', r'^transforms\..*'),

    # Shorthand alias used in docs for our own package
    (r'py:(class|mod|func|meth|obj|attr)', r'^pm\..*'),

    # Internal cross-refs to objects/attrs/methods that autosummary may not emit
    (r'py:(obj|attr|meth)', r'^pymovements\..*'),

    # Modules referenced in text but not importable via intersphinx targets
    (r'py:mod', r'^pymovements\.events(?:\.event_properties)?$'),

    # Custom exception names mentioned in text but not importable as a symbol
    (r'py:exc', r'^UnknownMeasure$'),
    (r'py:exc', r'^\.\.\s+deprecated:$'),


    # Matplotlib color types referenced in plotting API
    (r'py:class', r'^(?:colors\.Colormap|LinearSegmentedColormapType)$'),

    # Project-internal typing aliases used only in docs
    (r'py:class', r'^(?:ResourcesLike|DatasetDefinitionClass|SampleMeasure)$'),
    # Fully-qualified generic forms that appear in docstrings
    (
        r'py:class', r'^pymovements\.(?:dataset\.dataset_library\.'
        r'DatasetDefinitionClass|measure\.library\.SampleMeasure)$',
    ),

    # Fully-qualified references to our classes that aren't resolvable via intersphinx inventory
    (r'py:class', r'^pymovements\.dataset\.(?:Dataset|DatasetDefinition|DatasetPaths)$'),
    (r'py:class', r'^pymovements\.datasets\.Dataset$'),
    (r'py:class', r'^pymovements\.gaze\.Experiment$'),

    # Internal helper functions referenced in docs text
    (r'py:func', r'^(?:events\.engbert\.compute_threshold|_decompress)$'),

    # Residual autosummary cross-refs to attributes/methods on our high-level classes
    (r'py:(attr|meth)', r'^(?:Dataset|Gaze|DatasetPaths|Experiment)\..*'),
]


# -- Options for autosummary -------------------------------------------------
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False
add_module_names = True

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

html_theme_options = {
    'navigation_with_keys': False,
    'sidebar_includehidden': True,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/pymovements/pymovements',
            'icon': 'fa-brands fa-github',
        },
    ],
    'logo': {
        'image_light': 'https://raw.githubusercontent.com/pymovements/pymovements/main/docs/source/_static/logo.svg',  # noqa: E501
        'image_dark': 'https://raw.githubusercontent.com/pymovements/pymovements/main/docs/source/_static/logo.svg',  # noqa: E501
    },
}

# -- MyST configuration --------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html

myst_links_external_new_tab = True

nb_execution_timeout = 60
nb_execution_mode = 'auto'
nb_execution_show_tb = True

# -- Intersphinx options -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'polars': ('https://docs.pola.rs/api/python/stable', None),
    'feather': ('https://arrow.apache.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

# -- Options for favicons

favicons = [
    {'href': 'icon.svg'},
]


# -- Options for BibTeX ------------------------------------------------------
bibtex_bibfiles = ['bibliography.bib']
bibtex_default_style = 'author_year_style'
bibtex_reference_style = 'author_year'


class AuthorYearLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            yield f'{entry.persons["author"][0].rich_last_names[0]} et al., {entry.fields["year"]}'


class AuthorYearStyle(PlainStyle):
    default_label_style = AuthorYearLabelStyle


register_plugin('pybtex.style.formatting', 'author_year_style', AuthorYearStyle)


def getrev():
    try:
        revision = run(
            ['git', 'describe', '--tags', 'HEAD'],
            capture_output=True,
            check=True,
            text=True,
        ).stdout[:-1]
    except CalledProcessError:
        revision = 'main'

    return revision


REVISION = getrev()

extlinks = {
    'repo': (
        f'https://github.com/pymovements/pymovements/blob/{REVISION}/%s',
        '%s',
    ),
}

LINKCODE_URL = (
    f'https://github.com/pymovements/pymovements/blob/{REVISION}'
    '/src/pymovements/{filepath}#L{linestart}-L{linestop}'
)


# revised from https://gist.github.com/nlgranger/55ff2e7ff10c280731348a16d569cb73
def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    modname = info['module']
    topmodulename = modname.split('.')[0]
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        modpath = importlib.resources.files(topmodulename)
        filepath = os.path.relpath(inspect.getsourcefile(obj), modpath)
        if filepath is None:
            return
    except Exception:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return LINKCODE_URL.format(filepath=filepath, linestart=linestart, linestop=linestop)

# Copyright (c) 2022-2026 The pymovements Project Authors
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
import pkgutil
import string
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
copyright = '2022-2026 The pymovements Project Authors'
author = 'The pymovements Project Authors'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
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
    'myst_nb',  # load after `sphinx_mdinclude` to suppress extension error ('.md' registration)
]
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
    '.md': 'markdown',
}


def config_inited_handler(app, config):
    os.makedirs(os.path.join(app.srcdir, app.config.generated_path), exist_ok=True)


def doctree_resolved_handler(app, doctree, docname):
    """Open all external links in a new tab."""
    from docutils import nodes

    for node in doctree.traverse(nodes.reference):
        uri = node.get('refuri', '')
        if uri.startswith(('http://', 'https://')):
            node['target'] = '_blank'
            node['rel'] = 'noopener noreferrer'


def collect_property_members():
    """Collect public property names for every importable pymovements class path."""
    package = importlib.import_module('pymovements')
    modules = [package]
    for module_info in pkgutil.walk_packages(package.__path__, f'{package.__name__}.'):
        modules.append(importlib.import_module(module_info.name))

    property_members = {}
    for module in modules:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if not obj.__module__.startswith('pymovements'):
                continue
            properties = []
            for member_name in dir(obj):
                if member_name.startswith('_'):
                    continue
                member = inspect.getattr_static(obj, member_name, None)
                if not isinstance(member, property):
                    continue
                if getattr(member.fget, '__module__', '').startswith('pymovements'):
                    properties.append(member_name)
            if properties:
                property_members[f'{module.__name__}.{name}'] = properties
    return property_members


def _is_section_header(lines, index):
    """Check if the line at index starts a numpy-style docstring section."""
    return (
        index + 1 < len(lines)
        and bool(lines[index].strip())
        and not lines[index][0].isspace()
        and set(lines[index + 1].strip()) == {'-'}
    )


def strip_property_attributes_handler(app, what, name, obj, options, lines):
    """Remove property entries from the Attributes section of a class docstring.

    Properties are documented on their own pages (see ``_templates/property.rst``), which are
    the canonical cross-reference targets. Listing them in the class Attributes section as well
    would register duplicate object descriptions. Only the in-memory docstring lines are edited,
    the source docstrings stay complete for pydoclint. Must run before napoleon converts the
    numpy-style sections.
    """
    if what != 'class':
        return
    properties = set(app.config.autosummary_context['property_members'].get(name, []))
    if not properties:
        return

    start = next(
        (
            index for index in range(len(lines))
            if lines[index].strip() == 'Attributes' and _is_section_header(lines, index)
        ),
        None,
    )
    if start is None:
        return
    end = start + 2
    while end < len(lines) and not _is_section_header(lines, end):
        end += 1

    body = lines[start + 2:end]
    trailing_blank_count = 0
    while trailing_blank_count < len(body) and not body[-1 - trailing_blank_count].strip():
        trailing_blank_count += 1
    core = body[:len(body) - trailing_blank_count]

    # An entry is its unindented name line plus the following indented or blank lines.
    kept = []
    index = 0
    while index < len(core):
        entry_end = index + 1
        while entry_end < len(core) and not core[entry_end][:1].strip():
            entry_end += 1
        entry_name = core[index].split(':', 1)[0].strip()
        if entry_name not in properties:
            kept.extend(core[index:entry_end])
        index = entry_end
    while kept and not kept[-1].strip():
        kept.pop()

    if kept:
        lines[start + 2:end] = kept + body[len(body) - trailing_blank_count:]
    else:
        del lines[start:end]


def setup(app):
    app.add_config_value('REVISION', 'master', 'env')
    app.add_config_value('generated_path', '_generated', 'env')
    app.connect('config-inited', config_inited_handler)
    app.connect('doctree-resolved', doctree_resolved_handler)
    # napoleon connects with the default priority 500, lower priorities run first.
    app.connect('autodoc-process-docstring', strip_property_attributes_handler, priority=400)


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['pmep/TEMPLATE.md']
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

    # Matplotlib color types referenced in plotting API
    (
        r'py:class',
        r'^(?:colors\.Colormap|colors\.Normalize|LinearSegmentedColormapType|Normalize)$',
    ),

    # Project-internal typing aliases used only in docs
    (r'py:class', r'^(?:DatasetDefinitionClass|SampleMeasure)$'),
    # Fully-qualified generic forms that appear in docstrings
    (
        r'py:class', r'^pymovements\.(?:dataset\.dataset_library\.'
        r'DatasetDefinitionClass|measure\.samples\.library\.SampleMeasure)$',
    ),
    # generic types https://github.com/sphinx-doc/sphinx/issues/14159
    (r'py:class', r'.*dict\[str'),

    # Odd matplotlib reference seen in deprecated utils.plotting docs
    (r'py:class', r'^matplotlib\.pyplot\.figure$'),
]


# -- Options for autosummary -------------------------------------------------
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False
autosummary_context = {'property_members': collect_property_members()}
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
# Execute notebooks once and re-execute only when their content changes.
# The cache lives in docs/.jupyter_cache, so CI and Read the Docs still start fresh.
nb_execution_mode = 'cache'
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
bibtex_reference_style = 'label'


class AuthorYearLabelStyle(BaseLabelStyle):
    template: str = '{author} et al., {year}'

    def format_labels(self, sorted_entries):
        outputs: list[str] = []
        for entry in sorted_entries:
            candidate = self.template.format(
                author=entry.persons['author'][0].rich_last_names[0],
                year=entry.fields['year'],
            )

            if candidate in outputs:
                for suffix_char in string.ascii_lowercase:
                    suffix_candidate = candidate + suffix_char
                    if suffix_candidate not in outputs:
                        candidate = suffix_candidate
                        break
                else:
                    raise ValueError(f"character suffixes exhausted for '{candidate}'")

            outputs.append(candidate)
            yield candidate


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

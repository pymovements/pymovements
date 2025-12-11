===================
 Contributing Code
===================

Code of Conduct
---------------

At **pymovements**, we are commited to providing a welcoming, inclusive, and harassment-free
experience for everyone.

It is important to us that all participants treat one another’s person and contributions with
respect and dignity. Please read and follow our :ref:`code-of-conduct` for full details.

First-time Contributors
-----------------------

If you're looking for things to help with, try browsing our
`issue tracker <https://github.com/pymovements/pymovements/issues>`_ first. In particular, look for:

- `good first issues <https://github.com/pymovements/pymovements/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_
- `documentation issues <https://github.com/pymovements/pymovements/labels/documentation>`_

You do not need to ask for permission to work on any of these issues. The current status of the
issue will let you know if someone else is or was already working on it.

To get help fixing a specific issue, it's often best to comment on the issue itself. You're much
more likely to get targeted help if you provide details about what you've tried and where you've
looked.

To start out with developing, :ref:`install the dependencies <development-installation>` and
:ref:`create a branch <creating-a-branch>` for your contribution.

Create a :ref:`pull request <creating-pull-requests>` when you feel confident to publish your
progress. Don't hesitate if it's a work in progress, we can give you early feedback on your work.
If you can, try to add :ref:`tests <testing>` early on to verify correctness.

Getting Started
---------------

This is a general guide to contributing changes to pymovements.

Before you start developing, make sure to read our :ref:`documentation <startpage>` first.

.. _development-installation:

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

Make sure to install the latest pymovements version from the main branch.

```bash
git clone https://github.com/pymovements/pymovements.git
cd pymovements
pip install -e .
```

If you have a problem e.g. `command not found: pip`, check whether you have activated a virtual
environment.

.. _create-branch:

Creating a Branch
^^^^^^^^^^^^^^^^^

Before you start making changes to the code, create a local branch from the latest version of the
`main` branch.

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-branch
```

To shorten this call you can create a git alias via

```bash
git config alias.newb '!f() { git checkout main; git pull; git checkout -b $1; }; f'
```

You can then update main and create new branches with this command:

```bash
git newb your-new-branch
```

We do not allow for pushing directly to the `main` branch and merge changes exclusively by
:ref:`pull request <creating-pull-requests>`.

We will squash your commits into a single commit on merge to maintain a clean git history.
We use a linear git-history, where each commit contains a full feature/bug fix, such that each
commit represents an executable version. This way you also don't have to worry much about your
intermediate commits and can focus on getting your work done first.

.. _code-style:

Code Style
^^^^^^^^^^

We write our code to follow `PEP-8 <https://www.python.org/dev/peps/pep-0008)>`_ with a maximum
line-width of 100 characters. We additionally use type annotations as in
`PEP-484 <https://peps.python.org/pep-0484)>`_. For docstrings we use the
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ formatting standard.

We use `flake8 <https://pypi.org/project/flake8/>`_ for quick style checks,
`pylint <https://pypi.org/project/pylint/>`_ for thorough style checks and
`mypy <https://pypi.org/project/mypy/>`_ for checking type annotations.

You can check your code style by using `pre-commit <https://www.pre-commit.com>`_.
You can install `pre-commit` and `pylint` via pip.

**Note**: Quoting '.[dev]' ensures the command works in both bash and zsh.

```bash
pip install -e '.[dev]'
```

To always run style checks when pushing commits upstream,
you can register a pre-push hook by

```bash
pre-commit install --hook-type pre-push
```

If you want to run pre-commit for all your currently staged files, use

```bash
pre-commit
```

You can find the names of all defined hooks in the file `.pre-commit-config.yaml`.

If you want to run a specific hook you can use

```bash
pre-commit run mypy
pre-commit run pydocstyle
```

If you want to run a specific hook on a single file you can use

```bash
pre-commit run mypy --files src/pymovements/gaze/transforms.py
```

If you want to run all hooks on all git repository files use

```bash
pre-commit run -a
```

For running a specific hook on all git repository files use

```bash
pre-commit run mypy -a
```

.. _testing:

Testing
^^^^^^^

Tests are written using `pytest <https://docs.pytest.org>`_ and executed
in a separate environment using `tox <https://tox.readthedocs.io/en/latest/>`_.

If you have not yet installed `tox` and the testing dependencies you can do so via

```bash
pip install -e '.[dev]'
```

You can run all tests on all supported python versions run by simply calling `tox` in the
repository root.

```bash
tox
```

Running `tox` the first time in the repository will take a few minutes, as all necessary python
environments will have to be set up with their dependencies. Runtime should be short on the
subsequent runs.

If you add a new feature, please also include appropriate tests to verify its intended
functionality. We try to keep our code coverage close to 100%.

It is possible to limit the scope of testing to specific environments and files. For example, to
only test event-related functionality using the Python 3.10 environment use:

```bash
tox -e py310 -- tests/unit/events
```

In case you only want to run tests locally that do not require any network access you can use:

```bash
tox -e py310 -- -m "not network"
```

.. _documentation:

Documentation
^^^^^^^^^^^^^

Make sure to add docstrings to every class, method, and function that you add to the codebase.
Docstrings should include a description of all parameters, returns, and exceptions. Use the existing
documentation as an example.

To generate documentation pages, you can install the necessary dependencies using:

```bash
pip install -e '.[docs]'
```

`Sphinx <https://www.sphinx-doc.org>`_ generates the API documentation from the
numpydoc-style docstring of the respective modules/classes/functions.
You can build the documentation locally using the respective tox environment:

```bash
tox -e docs
```

It will appear in the `build/docs` directory.
Please note that in order to reproduce the documentation locally, you may need to install `pandoc`.
If necessary, please refer to the `installation guide <https://pandoc.org/installing.html>`_ for
detailed instructions.

To rebuild the full documentation use

```bash
tox -e docs -- -aE
```
.. _continuous-integration:

Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^

Tests, code style, and documentation are all additionally checked using a GitHub Actions
workflow which executes the appropriate tox environments. Merging of Pull requests will not be
possible until all checks pass.

.. _creating-pull-requests:

Creating Pull Requests
^^^^^^^^^^^^^^^^^^^^^^

Once you are ready to publish your changes:

- Create a `pull request (PR) <https://github.com/pymovements/pymovements/compare>`_.
- Provide a summary of the changes you are introducing, according to the default template.
- In case you are resolving an issue, remember to add a reference in the description.

The default template is meant as a helper and should guide you through the process of creating a
pull request. It's also totally fine to submit work in progress, in which case you'll likely be
asked to make some further changes.

If your change is a significant amount of work to write, we highly recommend starting by
opening an issue laying out what you want to do. That lets a conversation happen early in case other
contributors disagree with what you'd like to do or have ideas that will help you do it.

The best pull requests are focused, clearly describe what they're for and why they're correct, and
contain tests for whatever changes they make to the code's behavior. As a bonus, these are easiest
for someone to review, which helps your pull request get merged quickly. Standard advice about good
pull requests for open-source projects applies.

Do not squash your commits after you have submitted a pull request, as this
erases context during review. We will squash commits when the pull request is ready to be merged.

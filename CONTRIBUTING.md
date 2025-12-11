# Contributing to pymovements

Thank you for taking your time to contribute to pymovements!

We encourage you to report any bugs or contribute to new features, optimisations, or documentation.

Here we give you an overview of the workflow and best practices for contributing
to pymovements.

**Questions:** If you have any developer-related questions, please [open an issue](
https://github.com/pymovements/pymovements/issues/new/choose) or write us at
[pymovements@python.org](mailto:pymovements@python.org)

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Reporting Bugs](#reporting-bugs)
- [First-time Contributors](#first-time-contributors)
- [Getting Started](#getting-started)
    - [Development Installation](#development-installation)
    - [Creating a Branch](#creating-a-branch)
    - [Code Style](#code-style)
    - [Testing](#testing)
    - [Documentation](#documentation)
    - [Pull Requests](#pull-requests)
    - [Continuous Integration](#continuous-integration)
- [Core Developer Guidelines](#core-developer-guidelines)
- [Publishing Releases](#publishing-releases)
- [License](#license)
- [Questions](#questions)

## Code of Conduct

Everyone participating in the pymovements project, including in issues, pull requests,
discussions, and any other community spaces, is expected to treat others with respect
and to contribute to an open, welcoming, and inclusive environment.

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) for full details.

## Reporting Bugs

If you discover a bug, as a first step, please check the existing
[Issues](https://github.com/pymovements/pymovements/issues) to see if this bug has already been
reported.

In case the bug has not been reported yet, please do the following:

- [Open an issue](https://github.com/pymovements/pymovements/issues/new?labels=bug&template=ISSUE.md).
- Add a descriptive title to the issue and write a short summary of the problem.
- We provide you with a default template to guide you through a typical reporting process.
- Adding more context, including error messages and references to the problematic parts of the code,
  would be very helpful to us.

Once a bug is reported, our development team will try to address the issue as quickly as possible.

## First-time Contributors

If you're looking for things to help with, try browsing our [issue tracker](
https://github.com/pymovements/pymovements/issues) first. In particular, look for:

- [good first issues](https://github.com/pymovements/pymovements/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- [documentation issues](https://github.com/pymovements/pymovements/labels/documentation)

You do not need to ask for permission to work on any of these issues. The current status of the
issue will let you know if someone else is or was already working on it.

To get help fixing a specific issue, it's often best to comment on the issue itself. You're much
more likely to get targeted help if you provide details about what you've tried and where you've looked.

To start out with developing, [install the dependencies](#development-installation) and
[create a branch](#creating-a-branch) for your contribution.

Create a [pull request](#pull-requests) when you feel confident to publish your progress. Don't
hesitate if it's a work in progress, we can give you early feedback on your work.
If you can, try to add [unit tests](#testing) early on to verify correctness.

## Getting Started

This is a general guide to contributing changes to pymovements.

Before you start developing, make sure to read our [documentation](
https://pymovements.readthedocs.io/) first.

### Development Installation

Make sure to install the latest pymovements version from the main branch.

```bash
git clone https://github.com/pymovements/pymovements.git
cd pymovements
pip install -e .
```

If you have a problem e.g. `command not found: pip`, check whether you have activated a virtual environment.

### Creating a Branch

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
[pull requests](#pull-requests).

We will squash your commits into a single commit on merge to maintain a clean git history.
We use a linear git-history, where each commit contains a full feature/bug fix, such that each
commit represents an executable version. This way you also don't have to worry much about your
intermediate commits and can focus on getting your work done first.

### Code Style

We write our code to follow [PEP-8](https://www.python.org/dev/peps/pep-0008) with a maximum
line-width of 100 characters. We additionally use type annotations as in [PEP-484](
https://peps.python.org/pep-0484). For docstrings we use the [numpydoc](
https://numpydoc.readthedocs.io/en/latest/format.html) formatting standard.

We use [`flake8`](https://pypi.org/project/flake8/) for quick style checks and
[`pylint`](https://pypi.org/project/pylint/) for thorough style checks and [`mypy`](
https://pypi.org/project/mypy/) for checking type annotations.

You can check your code style by using [pre-commit](https://www.pre-commit.com).
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

### Testing

Tests are written using [Pytest](https://docs.pytest.org) and executed
in a separate environment using [Tox](https://tox.readthedocs.io/en/latest/).

If you have not yet installed `tox` and the testing dependencies you can do so via

```bash
pip install -e '.[dev]'
```

You can run all tests on all supported python versions run by simply calling `tox` in the repository root.

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

### Documentation

Make sure to add docstrings to every class, method, and function that you add to the codebase.
Docstrings should include a description of all parameters, returns, and exceptions. Use the existing
documentation as an example.
To generate documentation pages, you can install the necessary dependencies using:

```bash
pip install -e '.[docs]'
```

[Sphinx](https://www.sphinx-doc.org) generates the API documentation from the
numpydoc-style docstring of the respective modules/classes/functions.
You can build the documentation locally using the respective tox environment:

```bash
tox -e docs
```

It will appear in the `build/docs` directory.
Please note that in order to reproduce the documentation locally, you may need to install `pandoc`.
If necessary, please refer to the [installation guide](https://pandoc.org/installing.html) for
detailed instructions.

To rebuild the full documentation use

```bash
tox -e docs -- -aE
```

### Pull Requests

Once you are ready to publish your changes:

- Create a [pull request (PR)](https://github.com/pymovements/pymovements/compare).
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

### Continuous Integration

Tests, code style, and documentation are all additionally checked using a GitHub Actions
workflow which executes the appropriate tox environments. Merging of Pull requests will not be
possible until all checks pass.

## Core Developer Guidelines

Core developers should follow these rules when processing pull requests:

- Always wait for tests to pass before merging PRs.
- Use "[Squash and merge](https://github.com/blog/2141-squash-your-commits)" to merge PRs.
- Delete branches for merged PRs.
- Edit the final commit message before merging to conform to the [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) specification:

```
<type>[optional scope]: <description> (#PR-id)

- detailed description, wrapped at 72 characters
- bullet points or sentences are okay
- all changes should be documented and explained
- valid scopes are the names of the top-level directories in the package, like `dataset`, `gaze`, or `events`
```

Make sure:

- that when merging a multi-commit PR, the commit message doesn't
  contain the local history from the committer and the review history from
  the PR. Edit the message to only describe the end state of the PR.
- that the maximum subject line length is under 50 characters
- that the maximum line length of the commit message is under 72 characters
- to capitalize the subject and each paragraph.
- that the subject of the commit message has no trailing dot.
- to use the imperative mood in the subject line (e.g. "Fix typo in README").
- if the PR fixes an issue, that something like "Fixes #xxx." occurs in the body of the message
  (not in the subject).
- to use Markdown for formatting.

# Publishing Releases

Before releasing a new pymovements version make sure that all integration tests pass via `tox -e integration`.

You need to register an account on [PyPI](https://pypi.org/account/register/) and request maintainer privileges for releasing new pymovements versions.

The first step is releasing on GitHub. Our [release-drafter](https://github.com/pymovements/pymovements/blob/main/.github/release-drafter.yml) takes care of drafting a release log which should be
available on the [release page](https://github.com/pymovements/pymovements/releases). Please assign the listed PRs into the correct categories in the release draft. If all merged PRs adhered to
the [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) specification the release-drafter will have already taken care of this. Take special care for PRs that introduce breaking
changes. Specify the version tag according to the [Semantic Versioning 2.0.0](https://semver.org/) specification. After publishing the release on GitHub the latest commit will be tagged with the
specified version.

The next step is releasing pymovements on the PyPI repository.
This is currently done manually, so you need to run a `git pull` locally. It is recommended to use a separate local directory and not the one you use for development to make sure you are using a clean
source.

Now build a new package using

```
python -m build
```

This should result in two files being created in the `dist` directory: a `.whl` file and a `.tar.gz` file. The filenames should match the specified python version. If the filenames include the word
`dirty` then you need to make sure you work on a clean pymovements source. Your local files must not include any uncommited changes or files, otherwise your build will be flagged as dirty and will not
be adequate for uploading.

Now you can upload your `.whl` and `.tar.gz` files via

```
python -m twine upload dist/pymovements-${VERSION}*
```

Check that the [pymovements page](https://pypi.org/project/pymovements/) at the PyPI repository features the new pymovements version.

The next step is making sure the new version is uploaded into the conda-forge repository. This part is automated via the [pymovements-feedstock](https://github.com/conda-forge/pymovements-feedstock)
repository. A bot will create a PR and merge it after passing all tests. There might be issues when the new pymovements release includes changes in dependencies. You will then need to adjust the
`meta.yaml` found in the `recipe` directory.

## License

Please note that by contributing to the project, you agree that it will be licensed under the
[License](https://github.com/pymovements/pymovements/blob/main/LICENSE.txt) of this project.

If you did not write the code yourself, ensure the existing license is compatible and include the
license information in the contributed files, or obtain permission from the original author to
relicense the contributed code.

## Questions

If you have any developer-related questions, please [open an issue](
https://github.com/pymovements/pymovements/issues/new/choose) or write us at
[pymovements@python.org](mailto:pymovements@python.org)

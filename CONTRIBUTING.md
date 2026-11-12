# Contributing to pymovements

Thank you for taking your time to contribute to pymovements!

We encourage you to report any bugs or contribute to new features, optimisations, or documentation.

Here we give you an overview of the workflow and best practices for contributing
to pymovements.

**Questions:** If you have any developer-related questions, please [open an issue](
https://github.com/pymovements/pymovements/issues/new/choose) or write to us at
[pymovements@python.org](mailto:pymovements@python.org)

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Reporting Bugs](#reporting-bugs)
- [First-time Contributors](#first-time-contributors)
- [Getting Started](#getting-started)
    - [Development Installation](#development-installation)
    - [Creating a Branch](#creating-a-branch)
    - [Code Style](#code-style)
    - [Dependencies](#dependencies)
    - [Testing](#testing)
    - [Benchmarks](#benchmarks)
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

To shorten this call, you can create a git alias via

```bash
git config alias.newb '!f() { git checkout main; git pull; git checkout -b $1; }; f'
```

You can then update the main and create new branches with this command:

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

If you want to run a specific hook, you can use

```bash
pre-commit run mypy
pre-commit run pydocstyle
```

If you want to run a specific hook on a single file, you can use

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

### Dependencies

Every dependency in `pyproject.toml` has a lower bound and no upper bound. A new required
dependency needs a justification in the pull request description.

A package that only a few features need goes into an extra. Import it with
`import_optional_dependency` from `pymovements._utils.optional` on the code path that needs it.
The helper raises an `ImportError` that names the extra to install.

Heavy packages (scipy, matplotlib, pandas, scikit-learn, PIL) are imported inside the function that
uses them, so that `import pymovements` stays fast. Bind the module to an alias with
`importlib.import_module`, which keeps pylint's `import-outside-toplevel` check on. Imports
needed only for type annotations go under `TYPE_CHECKING`.

```python
plt = importlib.import_module('matplotlib.pyplot')
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
functionality. Our code coverage is at 100% and stays there, so every changed line needs a test
that runs it.

It is possible to limit the scope of testing to specific environments and files. For example, to
only test event-related functionality using the Python 3.11 environment use:

```bash
tox -e py311 -- tests/unit/events
```

In case you only want to run tests locally that do not require any network access, you can use:

```bash
tox -e py311 -- -m "not network"
```

### Benchmarks

Performance benchmarks live in `tests/benchmark` and use
[pytest-benchmark](https://pytest-benchmark.readthedocs.io). You can run them locally with

```bash
tox -e benchmark
```

The results are written to `benchmark-data.json` in the repository root.

On every pull request, a workflow benchmarks the head of your branch and its merge-base with `main`
on the same runner and fails if a benchmark gets more than 1.5 times slower. This comparison is
the authoritative regression check.

Every push to `main` adds the result of its head commit to the
[benchmark history chart](https://pymovements.github.io/pymovements/benchmarks/). The chart is
indicative only: its points come from shared GitHub runners of varying speed, so it shows host
speed as well as code speed.

### Documentation

Make sure to add docstrings to every class, method, and function that you add to the codebase.
Docstrings should include a description of all parameters, returns, and exceptions. Use the existing
documentation as an example.

Runnable examples in docstrings are executed as doctests by pytest. If an example writes files,
request the `doctest_tmp_cwd` fixture in a hidden `.. testsetup::` block; see its docstring in
`tests/doctest/doctest_fixtures.py` for details.

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

Once you are ready to publish your changes, create a
[pull request (PR)](https://github.com/pymovements/pymovements/compare). The default template
fills the body with the required sections. Every instruction in it is an HTML comment, so nothing
needs to be deleted before posting.

It's also totally fine to submit work in progress, in which case you'll likely be asked to make
some further changes.

If your change is a significant amount of work to write, we highly recommend starting by
opening an issue laying out what you want to do. That lets a conversation happen early in case other
contributors disagree with what you'd like to do or have ideas that will help you do it.

The best pull requests are focused, clearly describe what they're for and why they're correct, and
contain tests for whatever changes they make to the code's behavior. As a bonus, these are easiest
for someone to review, which helps your pull request get merged quickly. Standard advice about good
pull requests for open-source projects applies.

#### Title

The title starts with a prefix that names the type of change, followed by a colon, as in
`fix: handle empty gaze frames`. The prefix sets the label of the pull request and the section of
the release notes in which it appears. A scope in parentheses is optional, as in
`feat(events): ...`.

| Prefix | Label | Release notes section |
|---|---|---|
| `feat` | `enhancement` | ✨ Enhancements |
| `fix` | `fix` | 🐞 Bug Fixes |
| `docs` | `documentation` | 📖 Documentation |
| `dataset` | `dataset` | 📀 Datasets |
| `depr` | `deprecation` | ⚠️ Deprecations |
| `perf` | `performance` | 🚀 Performance improvements |
| `build` | `build` | 🛠️ Maintenance |
| `chore`, `ci`, `refactor`, `test` | `internal` | 🛠️ Maintenance |
| `release` | `release` | no section of its own, used by maintainers for release pull requests |

Add `!` after prefix and scope for a breaking change, as in `feat!: ...` or
`feat(events)!: ...`. This sets the `breaking` label, lists the pull request under
💥 Breaking Changes and bumps the minor version of the next release.

#### Body

Three sections are always present, in this order:

- **Description** states what the change does and why. If the pull request fixes an issue, it
  contains `Resolves #N` so that the issue closes on merge. Issues the pull request relates to
  without resolving them are mentioned here as well.
- **Changes** is a short bullet list that maps the diff, one bullet per change.
- **Testing** states how the change was verified: the tests added or run and any manual checks.

Four sections are added on demand. ⚠️ Deprecation and 💥 Breaking Change open the body in that
order, before Description, since readers of the release notes look for them first. 🚨 Merge Danger
and Dependencies follow Testing, in that order.

- **⚠️ Deprecation** has one paragraph per deprecated item, naming the item, the version that
  deprecates it, its replacement and the removal version. The pull request carries the
  `deprecation` label. A `depr` title sets it. A `feat` pull request that also deprecates
  something gets it from a maintainer, next to `enhancement`.
- **💥 Breaking Change** has one paragraph per break, naming the behavior or signature that
  changes, the code that stops working and what to use instead. The title carries `!`.
- **🚨 Merge Danger** marks a change that is hard to reverse after a merge and is not an API break.
  Most merges are two-way doors, since a revert restores the previous state. A persisted file
  format, a dataset source or checksum, release tooling, CI tooling or a dependency floor is a
  one-way door, since a revert does not recall files already written, data already downloaded or a
  release already published, and does not restore environments already resolved against a floor.
  The section names the one-way door (what cannot be undone after the merge) and the blast radius
  (who and what is affected).
- **Dependencies** lists the pull requests this one requires, the pull requests that require it and
  the pull requests it conflicts with, so that the merge order is visible.

The body closes with the checklist from the template. It is a self-check for the author before
requesting a review, not evidence for the reviewer: verification lives in Testing. First-time
contributors add themselves to `CITATION.cff`.

Do not squash your commits after you have submitted a pull request, as this
erases context during review. We will squash commits when the pull request is ready to be merged.

Reviewers check pull requests against our
[Review Guidelines](https://pymovements.readthedocs.io/en/latest/contributing/review-guidelines.html)
(`docs/source/contributing/review-guidelines.rst` in the repository).

When a review requests changes, a maintainer converts your pull request to a draft. Once you
have addressed the findings, mark it ready for review again so that it shows up for the reviewers.

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
the [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/) specification, the release-drafter will have already taken care of this. Take special care for PRs that introduce breaking
changes. Specify the version tag according to the [Semantic Versioning 2.0.0](https://semver.org/) specification. After publishing the release on GitHub, the latest commit will be tagged with the
specified version.

The next step is releasing pymovements on the PyPI repository.
This is currently done manually, so you need to run a `git pull` locally. It is recommended to use a separate local directory and not the one you use for development to make sure you are using a clean
source.

Now build a new package using

```
python -m build
```

This should result in two files being created in the `dist` directory: a `.whl` file and a `.tar.gz` file. The filenames should match the specified python version. If the filenames include the word
`dirty` then you need to make sure you work on a clean pymovements source. Your local files must not include any uncommitted changes or files, otherwise your build will be flagged as dirty and will
not
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
https://github.com/pymovements/pymovements/issues/new/choose) or write to us at
[pymovements@python.org](mailto:pymovements@python.org)

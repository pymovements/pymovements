=======================
 Maintainer Guidelines
=======================

Merging Pull Requests
---------------------

Maintainers should follow these rules when processing pull requests:

- Always wait for tests to pass before merging PRs.
- Use "`Squash and merge <https://github.com/blog/2141-squash-your-commits>`_" to merge PRs.
- Delete branches for merged PRs.
- Edit the final commit message before merging to conform to the
  `Conventional Commit <https://www.conventionalcommits.org/en/v1.0.0/>`_ specification.

.. code::

    <type>[optional scope]: <description> (#PR-id)

    - detailed description, wrapped at 72 characters
    - bullet points or sentences are okay
    - all changes should be documented and explained
    - valid scopes are the names of main package modules, like `dataset`, `gaze`, or `plotting`

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

Publishing Releases
-------------------

Before releasing a new pymovements version make sure that all integration tests pass via
`tox -e integration`.

You need to register an account on `PyPI <https://pypi.org/account/register/>`_ and request
maintainer privileges for releasing new pymovements versions.

The first step is releasing on GitHub. Our
`release-drafter <https://github.com/pymovements/pymovements/blob/main/.github/release-drafter.yml>`_
takes care of drafting a release log which should be available on the
`release page <https://github.com/pymovements/pymovements/releases>`_. Please assign the listed PRs
into the correct categories in the release draft. If all merged PRs adhered to the
`Conventional Commit <https://www.conventionalcommits.org/en/v1.0.0/>`_ specification the
release-drafter will have already taken care of this. Take special care for PRs that introduce
breaking changes. Specify the version tag according to the
`Semantic Versioning 2.0.0 <https://semver.org/>`_ specification. After publishing the release on
GitHub the latest commit will be tagged with the specified version.

Check that the `pymovements page <https://pypi.org/project/pymovements/>`_ at the PyPI repository
features the new pymovements version.

The next step is making sure the new version is uploaded into the conda-forge repository. This part
is automated via the `pymovements-feedstock <https://github.com/conda-forge/pymovements-feedstock>`_
repository. A bot will create a PR and merge it after passing all tests. There might be issues when
the new pymovements release includes changes in dependencies. You will then need to adjust the
`meta.yaml` found in the `recipe` directory.

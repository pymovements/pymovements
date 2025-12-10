=======================
 Maintainer Guidelines
=======================

Merging Pull Requests
---------------------

Maintainers should follow these rules when processing pull requests:

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

Publishing Releases
-------------------

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

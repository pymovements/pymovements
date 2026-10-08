===================
 Review Guidelines
===================

These guidelines describe what we check when reviewing a pull request to pymovements. They apply
to reviews by maintainers and contributors alike, and they make a good checklist before you open a
pull request yourself.

pymovements is scientific software used by research labs. Silently wrong numbers are the worst
failure mode, worse than a crash. When a claim about correctness can be checked against the math,
the code or a hand-computed example, check it. Many of our users are researchers rather than
software engineers, so a clear API and good docstrings carry real weight.

Many pull requests come from students and first-time contributors. Review firmly on substance and
explain the reason behind a convention, so that the contributor learns the rule and not only the
fix. First-time contributors can add themselves to ``CITATION.cff`` before the merge to be
credited in the next release.

Before You Start
----------------

- Review the current state of the pull request. Fetch the latest head before you begin.
- Check CI first (``gh pr checks <N>``). Spend no findings on failures CI already reports, but
  explain the cause of a confusing failure if the diff shows it.
- Read the existing review threads. Verify earlier findings instead of raising them again, and
  report a finding that came back as a regression.
- Judgment calls that an earlier review round accepted (naming, structure, style) stay settled
  unless there is new evidence.

Maintainer Commits
------------------

Maintainers often add commits on top of a contributor's work to make a pull request ready for
merging. Check the commit authors first and review the two sets separately: the contributor's
changes first, then the maintainer's changes, opened by a short summary of what they did. The
summary lets the contributor see what was changed on top of their work.

A maintainer cannot approve their own commits. If these changes are substantial (rework, new
helpers, behavior changes, as opposed to typo, lint or docs fixes), another maintainer reviews
them.

Numerical and Algorithmic Changes
---------------------------------

- Check units and coordinate conventions (pixels or degrees of visual angle, screen origin).
- Check whether an operation applies per trial or stimulus, or globally, when trial columns exist.
- Require at least one test with a known input and an expected output that can be verified by
  hand. A test that only shows the code runs does not establish that an algorithm is correct.
- The expected value of a test is an independent literal or hand computation. If it is built
  through the same code path as the implementation (the same rounding, the same helper), the test
  hides the error it should catch.

Edge Cases in Gaze and Event Code
---------------------------------

Look for empty frames, all-NaN or partly NaN samples, a single row or fixation, monocular and
binocular column layouts, several trials or stimuli in one frame, and missing optional columns.
An unhandled case fails loudly with a clear error message instead of returning wrong values.

Functions leave arrays, dictionaries and DataFrames passed in by the caller unchanged. They copy
the input or build a new object, and they work regardless of the key order of a dictionary.

API Consistency
---------------

- Naming, signature style, and whether a method modifies in place or returns a new object match
  the neighbouring methods.
- New public API is exported in the relevant ``__init__.py``.
- Conversion or validation logic used in several places lives in one shared helper, so the copies
  cannot drift apart.
- ``@overload`` signatures actually narrow the type under mypy.
- A change in behavior of an existing signature needs a justification. Could an existing call
  behave differently?
- Deprecations follow the project cycle: a warning in the next minor release, removal five minor
  releases later (for example deprecated in v0.28.0, removed in v0.33.0). The warning names the
  removal version, and a test using ``assert_deprecation_is_removed`` covers it. The pull request
  description has a "Deprecation" section and carries the ``deprecation`` label.

Tests
-----

- Patch coverage is 100 %. Maintainers do not merge below that. Codecov reports the coverage of
  the pushed head, so for commits that are not pushed yet, map each changed line, error branches
  included, to a test that runs it.
- Tests go through the public API. A new test that calls an underscore-prefixed function is
  rewritten to reach the same code path through the public entry point, mocking at the external
  boundary (network, file system) rather than at the private helper.
- Test functions are free of ``if``/``else``. Each branch becomes its own
  ``pytest.mark.parametrize`` case with an explicit expected value, or its own test function.
  Fixtures may contain logic, and complex fixture logic gets its own tests in
  ``tests/fixtures/<name>_fixtures_test.py``.
- Error tests assert the error message, not only the exception type.
- New lines are ideally covered without ``tests/unit/dataset/dataset_test.py``, which is slow. A
  line that only this module covers wants a targeted test. When ``dataset_test.py`` catches a
  regression, add a fast unit test that pins it as well.
- Integration tests in ``tests/integration/`` download real datasets. Run them only for a dataset
  whose ``sources`` the pull request changes, and only for that dataset.

Documentation
-------------

- Linters already check the numpydoc structure and the completeness of Parameters, Returns and
  Raises. Review what they cannot check: user-facing API has a usage example, and docstrings state
  units and coordinate conventions where they matter.
- A new public property has its own entry in the ``Attributes`` section of the class docstring, in
  definition order.
- User-visible changes update the pages under ``docs/source/``.
- Changes to ``CITATION.cff``: every author has an ``orcid``. The maintainer block comes first in
  its fixed order, then contributors alphabetically by family name and given names, and the last
  entry is always Jäger, Lena A. A common mistake is a new entry appended at the very end.

polars
------

- Where a row-wise ``map_elements`` call or a Python loop could be a vectorized expression, name
  the expression.
- Watch for silent dtype changes, especially in time columns.

Severity
--------

- **Blocker**: wrong results, corrupted fixtures, a broken public API, a behavior change without
  justification, a new algorithm without a hand-verifiable test, patch coverage below 100 %.
- **Should-fix**: missing edge-case tests, missing documentation, error messages that are not
  asserted, avoidable row-wise polars operations.
- **Nit**: style points no linter enforces. What a linter enforces is not worth a comment.

Mark uncertainty as uncertainty: a wrong confident comment costs more than an honest question. A
review that finds nothing says so plainly.

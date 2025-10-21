# Copyright (c) 2025 The pymovements Project Authors
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
"""Provide fixtures to assert deprecations."""
import re

import pytest

from pymovements import __version__


@pytest.fixture(name='assert_deprecation_is_removed', scope='session')
def fixture_assert_deprecation_is_removed():
    """Assert that function deprecation is removed as scheduled."""
    regex = re.compile(r'.*will be removed in v(?P<version>[0-9]+[.][0-9]+[.][0-9]+)[.)].*')

    def _assert_deprecation_is_removed(
            *,
            function_name: str,
            warning_message: str,
            scheduled_version: str,
            current_version: str | None = None,
    ) -> None:
        """Assert that function deprecation is removed as scheduled.

        Parameters
        ----------
        function_name: str
            Name of the deprecated functionality.
        warning_message: str
            The DeprecationWarning message.
        scheduled_version: str
            The version that is scheduled for removal of the deprecated feature (without leading v).
        current_version: str | None
            The current version without the leading v. If ``None``, use ``pymovements.__version__``.
            (default: None)

        Raises
        ------
        AssertionError
            If the warning message does not match the expected regex, or if the scheduled version
            from the warning message does not match the expected scheduled version, or if the
            current version is equal or above the version that was scheduled for removal of the
            deprecated feature.

        """
        if current_version is None:
            current_version = __version__

        match = regex.match(warning_message)

        if not match:
            raise AssertionError(
                'DeprecationWarning message does not match regex.\n'
                f'message: {warning_message}\n'
                f'regex: {regex.pattern}',
            )

        parsed_scheduled_version = match.groupdict()['version']

        if parsed_scheduled_version != scheduled_version:
            raise AssertionError(
                'scheduled version from warning message does not match expected version.\n'
                f'parsed scheduled version: {parsed_scheduled_version}\n'
                f'expected scheduled version: {scheduled_version}\n',
            )

        current_version_stem = current_version.split('+')[0].split('-')[0]

        assert current_version_stem < scheduled_version, (
            f'{function_name} was scheduled to be removed in v{scheduled_version}. '
            f'Current version is v{current_version}.'
        )
    return _assert_deprecation_is_removed

# Copyright (c) 2025-2026 The pymovements Project Authors
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
"""Test deprecation fixtures."""

import re

import pytest

from pymovements import __version__


@pytest.mark.parametrize(
    (
        'function_name',
        'warning_message',
        'scheduled_version',
        'current_version',
        'assertion_message',
    ),
    [
        pytest.param(
            'curly_to_regex',
            'Call to deprecated function (or staticmethod) curly_to_regex. '
            '-- Deprecated since version v0.21.1.',
            '0.26.1',
            '0.26.0',
            'DeprecationWarning message does not match regex.',
            id='not_scheduled_for_removal',
        ),
        pytest.param(
            'curly_to_regex',
            'Call to deprecated function (or staticmethod) curly_to_regex. '
            '(This module will be removed in v0.27.0.) -- Deprecated since version v0.21.1.',
            '0.26.0',
            '0.25.0',
            'scheduled version from warning message does not match expected version',
            id='scheduled_version_wrong',
        ),
        pytest.param(
            'curly_to_regex',
            'Call to deprecated function (or staticmethod) curly_to_regex. '
            '(This module will be removed in v0.26.0.) -- Deprecated since version v0.21.1.',
            '0.26.0',
            '0.26.0',
            'curly_to_regex was scheduled to be removed in v0.26.0. Current version is v0.26.0.',
            id='current_version_is_scheduled_version',
        ),
        pytest.param(
            'curly_to_regex',
            'Call to deprecated function (or staticmethod) curly_to_regex. '
            '(This module will be removed in v0.26.0.) -- Deprecated since version v0.21.1.',
            '0.26.0',
            '0.26.0+16.g7c26d6e1',
            'curly_to_regex was scheduled to be removed in v0.26.0. Current version is v0.26.0.16.',
            id='scheduled_version_post_commit',
        ),
        pytest.param(
            'curly_to_regex',
            'Call to deprecated function (or staticmethod) curly_to_regex. '
            '(This module will be removed in v0.26.0.) -- Deprecated since version v0.21.1.',
            '0.26.0',
            '0.26.0+16.g7c26d6e1.dirty',
            'curly_to_regex was scheduled to be removed in v0.26.0. Current version is v0.26.0.*di',
            id='scheduled_version_dirty',
        ),
        pytest.param(
            'curly_to_regex',
            'Call to deprecated function (or staticmethod) curly_to_regex. '
            '(This module will be removed in v0.26.0.) -- Deprecated since version v0.21.1.',
            '0.26.0',
            '0.26.0-rc.1',
            'curly_to_regex was scheduled to be removed in v0.26.0. Current version is v0.26.0-rc.',
            id='scheduled_version_rc',
        ),
        pytest.param(
            'curly_to_regex',
            'Call to deprecated function (or staticmethod) curly_to_regex. '
            '(This module will be removed in v0.26.0.) -- Deprecated since version v0.21.1.',
            '0.26.0',
            '0.26.1',
            'curly_to_regex was scheduled to be removed in v0.26.0. Current version is v0.26.1.',
            id='patch_after_scheduled_version',
        ),
        pytest.param(
            'curly_to_regex',
            'Call to deprecated function (or staticmethod) curly_to_regex. '
            '(This module will be removed in v0.26.0.) -- Deprecated since version v0.21.1.',
            '0.26.0',
            '0.27.0',
            'curly_to_regex was scheduled to be removed in v0.26.0. Current version is v0.27.0.',
            id='minor_after_scheduled_version',
        ),
    ],
)
def test_assert_deprecation_fixture_assert_false(
    function_name,
    warning_message,
    scheduled_version,
    current_version,
    assertion_message,
    assert_deprecation_is_removed,
):
    with pytest.raises(AssertionError, match=assertion_message):
        assert_deprecation_is_removed(
            function_name=function_name,
            warning_message=warning_message,
            scheduled_version=scheduled_version,
            current_version=current_version,
        )


def test_assert_deprecation_fixture_default_current_version_assert_false(
    assert_deprecation_is_removed,
):
    base_version_regex = re.compile(r'(\d+[.]\d+[.]\d+)([+]?[-]?[a-z]?)?')
    scheduled_version = base_version_regex.match(__version__).group(1)
    warning_message = f'(This module will be removed in v{scheduled_version}.)'

    assertion_message = (
        f'scheduled .* removed in v{scheduled_version}. Current version is v{scheduled_version}'
    )

    with pytest.raises(AssertionError, match=assertion_message):
        assert_deprecation_is_removed(
            function_name='my_function',
            warning_message=warning_message,
            scheduled_version=scheduled_version,
        )


@pytest.mark.parametrize(
    ('function_name', 'warning_message', 'scheduled_version', 'current_version'),
    [
        pytest.param(
            'curly_to_regex',
            'Call to deprecated function (or staticmethod) curly_to_regex. '
            '(This module will be removed in v0.26.0.) -- Deprecated since version v0.21.1.',
            '0.26.0',
            '0.24.0',
            id='minor_before_scheduled_version',
        ),
        pytest.param(
            'curly_to_regex',
            'Call to deprecated function (or staticmethod) curly_to_regex. '
            '(This module will be removed in v0.26.0.) -- Deprecated since version v0.21.1.',
            '0.26.0',
            '0.25.9',
            id='patch_before_scheduled_version',
        ),
        pytest.param(
            'curly_to_regex',
            'Call to deprecated function (or staticmethod) curly_to_regex. '
            '(This module will be removed in v0.26.0.) -- Deprecated since version v0.21.1.',
            '0.26.0',
            '0.25.5+16.g7c26d6e1.dirty',
            id='patch_before_scheduled_version_dirty',
        ),
    ],
)
def test_assert_deprecation_fixture_assert_true(
    function_name,
    warning_message,
    scheduled_version,
    current_version,
    assert_deprecation_is_removed,
):
    assert_deprecation_is_removed(
        function_name=function_name,
        warning_message=warning_message,
        scheduled_version=scheduled_version,
        current_version=current_version,
    )

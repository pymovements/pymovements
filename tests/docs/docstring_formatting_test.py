# Copyright (c) 2026 The pymovements Project Authors
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
"""Test docstring formatting."""
import pytest


def test_deprecated_directives_are_separated_and_indented(request: pytest.FixtureRequest) -> None:
    """Require valid reStructuredText formatting for deprecation directives."""
    source_directory = request.config.rootpath / 'src' / 'pymovements'
    violations = []

    for source_file in sorted(source_directory.rglob('*.py')):
        lines = source_file.read_text(encoding='utf-8').splitlines()
        for line_number, line in enumerate(lines):
            if '.. deprecated::' not in line:
                continue

            directive_indent = len(line) - len(line.lstrip())
            has_blank_line = line_number > 0 and not lines[line_number - 1].strip()
            next_line = lines[line_number + 1] if line_number + 1 < len(lines) else ''
            content_indent = len(next_line) - len(next_line.lstrip())
            if not has_blank_line or content_indent <= directive_indent:
                violations.append(
                    f'{source_file.relative_to(source_directory)}:{line_number + 1}',
                )

    assert not violations, 'malformed deprecation directives: ' + ', '.join(violations)

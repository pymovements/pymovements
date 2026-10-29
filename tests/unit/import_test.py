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
"""Test which third-party packages ``import pymovements`` loads."""
import subprocess
import sys

import pytest


@pytest.fixture(name='imported_modules', scope='module')
def fixture_imported_modules():
    code = 'import sys; import pymovements; print("\\n".join(sys.modules))'
    result = subprocess.run(
        [sys.executable, '-c', code], check=True, capture_output=True, text=True,
    )
    return set(result.stdout.splitlines())


@pytest.mark.parametrize(
    'module',
    [
        pytest.param('fastexcel', id='fastexcel'),
        pytest.param('matplotlib', id='matplotlib'),
        pytest.param('pandas', id='pandas'),
        pytest.param('PIL', id='PIL'),
        pytest.param('pyreadr', id='pyreadr'),
        pytest.param('scipy.optimize', id='scipy.optimize'),
        pytest.param('scipy.signal', id='scipy.signal'),
        pytest.param('scipy.stats', id='scipy.stats'),
        pytest.param('sklearn', id='sklearn'),
    ],
)
def test_import_pymovements_does_not_load_module(imported_modules, module):
    assert module not in imported_modules

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
"""Test the public interface of pymovements.synthetic."""
import pytest

import pymovements as pm
from pymovements import synthetic


@pytest.mark.parametrize(
    'name',
    [
        pytest.param('EventGenerator', id='EventGenerator'),
        pytest.param('EventGeneratorLibrary', id='EventGeneratorLibrary'),
        pytest.param('SampleGenerator', id='SampleGenerator'),
        pytest.param('SampleGeneratorLibrary', id='SampleGeneratorLibrary'),
        pytest.param('generate_events', id='generate_events'),
        pytest.param('generate_samples', id='generate_samples'),
        pytest.param('register_generator', id='register_generator'),
        pytest.param('step_function', id='step_function'),
    ],
)
def test_synthetic_exports_public_name(name):
    assert name in synthetic.__all__
    assert hasattr(synthetic, name)


def test_all_names_are_available_attributes():
    missing = [name for name in synthetic.__all__ if not hasattr(synthetic, name)]
    assert not missing, f'names in __all__ without attribute: {missing}'


def test_synthetic_is_exported_at_package_root():
    assert 'synthetic' in pm.__all__
    assert pm.synthetic is synthetic

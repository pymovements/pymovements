# Copyright (c) 2023-2025 The pymovements Project Authors
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
"""Test pymovements paths."""
import pytest
import yaml

from pymovements.dataset._utils._yaml import type_constructor


def test_type_constructor_assertion_error(make_text_file):
    yaml_content = """\
    !test
    """
    yaml.add_multi_constructor('!', type_constructor, yaml.SafeLoader)
    yaml_file = make_text_file(filename='test_yaml_file', body=yaml_content)
    with pytest.raises(ValueError) as excinfo:
        with open(yaml_file, encoding='utf-8') as f:
            yaml.safe_load(f)
    msg, = excinfo.value.args
    assert msg == "Unknown node=ScalarNode(tag='!test', value='')"


def test_module_name_not_found_error(make_text_file):
    yaml_content = """\
    !pm.notexisting
    """
    yaml.add_multi_constructor('!', type_constructor, yaml.SafeLoader)
    yaml_file = make_text_file(filename='test_yaml_file', body=yaml_content)
    with pytest.raises(ModuleNotFoundError) as excinfo:
        with open(yaml_file, encoding='utf-8') as f:
            yaml.safe_load(f)
    msg, = excinfo.value.args
    assert msg == "No module named 'pm'"


def test_unknown_attribute_error(make_text_file):
    yaml_content = """\
    !yaml.notexisting
    """
    yaml.add_multi_constructor('!', type_constructor, yaml.SafeLoader)
    yaml_file = make_text_file(filename='test_yaml_file', body=yaml_content)
    with pytest.raises(ValueError) as excinfo:
        with open(yaml_file, encoding='utf-8') as f:
            yaml.safe_load(f)
    msg, = excinfo.value.args
    assert msg == 'Unknown type: notexisting for module yaml'

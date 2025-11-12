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
"""Yaml utilities."""
from __future__ import annotations

import builtins
import importlib
from typing import Any

import yaml


# generalized constructor for !* tags
def type_constructor(
        loader: yaml.SafeLoader,
        prefix: str,
        node: yaml.Node,
) -> type:
    """Resolve a YAML tag to a corresponding Python type.

    This function is used to handle custom YAML tags (e.g., `!polars.Int64`)
    by mapping the tag to a Python type or class name. The type name is
    extracted from the YAML tag and evaluated to return the corresponding
    Python object. If the type cannot be resolved, an error is raised.

    Parameters
    ----------
    loader: yaml.SafeLoader
        The YAML loader being used to parse the YAML document.
    prefix: str
        A string prefix for the custom tag (e.g., '!').
    node: yaml.Node
        The YAML node containing the tag and associated value.

    Returns
    -------
    type
        The Python type or class corresponding to the YAML tag.

    Raises
    ------
    ValueError: If the specified type name in the tag cannot be resolved
                to a valid Python object.

    Example:
        # Example YAML document:
        # !polars.Int64
        #
        # Resolves to the Python type `pl.Int64` (assuming `pl` is a valid module).

    """
    # pylint: disable=unused-argument
    # extract the type name (e.g., from !polars.Int64 to polars.Int64)
    type_name = node.tag[1:]

    built_in_types = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict,
        'set': set,
    }

    # check for built-in types first
    if type_name in built_in_types:
        return built_in_types[type_name]

    try:
        module_name, type_attr = type_name.rsplit('.', 1)
        module = __import__(module_name)
        return getattr(module, type_attr)

    # module does not have this file type
    except AttributeError as exc:
        raise ValueError(
            f'Unknown type: {type_attr} for module {module_name}',
        ) from exc
    except ValueError as exc:
        raise ValueError(f'Unknown {node=}') from exc


def substitute_types(data: Any) -> Any:
    """Substitute types in data with !-prefixed strings."""
    if isinstance(data, dict):
        return {k: substitute_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [substitute_types(v) for v in data]
    if isinstance(data, type):
        if data.__module__ == 'builtins':
            return f'!{data.__name__}'
        return f'!{data.__module__}.{data.__name__}'
    return data


def reverse_substitute_types(data: Any) -> Any:
    """Substitute !-prefixed strings with python types."""
    if isinstance(data, dict):
        return {k: reverse_substitute_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [reverse_substitute_types(v) for v in data]
    if isinstance(data, str) and data.startswith('!'):
        type_name = data[1:]
        if '.' in type_name:
            module_name, class_name = type_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        return getattr(builtins, type_name)
    return data

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
"""Helpers for deprecations."""
from __future__ import annotations

from typing import Any
from warnings import warn


class DeprecatedMetaClass(type):
    """MetaClass for deprecated class aliases.

    The class serves as an equivalent alias for the `isinstance()` and `issubclass()` methods.
    It supports subclassing of the deprecated class.

    Examples
    --------
    This is how a deprecated alias is defined:
    >>> class NewClass:
    ...     variable = 42
    >>>
    >>> class OldClass(metaclass=DeprecatedMetaClass):
    ...     _DeprecatedMetaClass__alias = NewClass
    ...     _DeprecatedMetaClass__version_deprecated = 'v1.23.4'
    ...     _DeprecatedMetaClass__version_removed = 'v2.0.0'

    Instantiating `OldClass` gives a warning:
    >>> old_class = OldClass()  # doctest: +SKIP
    DeprecationWarning('OldClass has been renamed to NewClass in v1.23.4 and will be removed in
      v2.0.0.')

    As you see, an `OldClass` object is an instance of both OldClass and NewClass:
    >>> isinstance(old_class, NewClass)  # doctest: +SKIP
    True

    As well as vice versa:
    >>> isinstance(NewClass(), OldClass)
    True
    """

    def __new__(
            mcs: type[DeprecatedMetaClass],
            name: str,
            bases: tuple,
            classdict: dict,
            *args: Any,
            **kwargs: Any,
    ) -> DeprecatedMetaClass:
        """Create new deprecated class."""
        alias = classdict.get('_DeprecatedMetaClass__alias')
        version_deprecated = classdict.get('_DeprecatedMetaClass__version_deprecated')
        version_removed = classdict.get('_DeprecatedMetaClass__version_removed')

        if alias is not None:
            def new(cls: type, *args: Any, **kwargs: Any) -> type:
                alias = getattr(cls, '_DeprecatedMetaClass__alias')
                version_deprecated = getattr(cls, '_DeprecatedMetaClass__version_deprecated')
                version_removed = getattr(cls, '_DeprecatedMetaClass__version_removed')

                warn(
                    f"{cls.__name__} has been renamed to {alias.__name__} "
                    f"in {version_deprecated} "
                    f"and will be removed in {version_removed}.",
                    DeprecationWarning, stacklevel=2,
                )

                return alias(*args, **kwargs)

            classdict['__new__'] = new
            classdict['_DeprecatedMetaClass__alias'] = alias
            classdict['_DeprecatedMetaClass__version_deprecated'] = version_deprecated
            classdict['_DeprecatedMetaClass__version_removed'] = version_removed

        fixed_bases = set()

        for b in bases:
            alias = getattr(b, '_DeprecatedMetaClass__alias', None)
            version_deprecated = classdict.get('_DeprecatedMetaClass__version_deprecated')
            version_removed = classdict.get('_DeprecatedMetaClass__version_removed')

            if alias is not None:
                warn(
                    f"{mcs.__name__} has been renamed to {alias.__name__} "
                    f"in {version_deprecated} "
                    f"and will be removed in {version_removed}.",
                    DeprecationWarning, stacklevel=2,
                )

            fixed_bases.add(alias or b)

        return super().__new__(mcs, name, tuple(fixed_bases), classdict, *args, **kwargs)

    def __subclasscheck__(cls, subclass: Any) -> bool:
        """Check if is subclass of deprecated class.

        Provides implementation for issubclass().
        """
        if subclass is cls:
            return True
        return issubclass(subclass, getattr(cls, '_DeprecatedMetaClass__alias'))

    def __instancecheck__(cls, instance: Any) -> bool:
        """Check if is instance of deprecated class.

        Provides implementation for isinstance().
        """
        # pylint: disable=no-value-for-parameter
        # pylint doesn't get that this is a metaclass method:
        # see: https://github.com/pylint-dev/pylint/issues/3268
        return any(cls.__subclasscheck__(c) for c in (type(instance), instance.__class__))

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
"""ResourceDefinitions and ResourceDefinition module."""
from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from warnings import warn

from deprecated.sphinx import deprecated

from pymovements._utils._html import repr_html


@repr_html()
@dataclass
class ResourceDefinition:
    """ResourceDefinition definition.

    Attributes
    ----------
    content: str
        The content type of the resource.
    filename: str | None
        The target filename of the downloadable resource. This may be an archive. (default: None)
    url: str | None
        The URL to the downloadable resource. (default: None)
    mirrors: list[str] | None
        An optional list of additional mirror URLs to download the resource. If downloading the
        resource from :py:attr:`~pymovements.ResourceDefinition.url` fails, these mirror URLs are
        used in order of appearance. (default: None)
    md5: str | None
        The MD5 checksum of the downloadable resource. (default: None)
    filename_pattern: str | None
        The filename pattern of the resource files. Named groups will
        be parsed as metadata will appear in the `fileinfo` dataframe. (default: None)
    filename_pattern_schema_overrides: dict[str, type] | None
        If named groups are present in the `filename_pattern`, this specifies their particular
        datatypes. (default: None)
    load_function: str | None
        The name of the function used to load the data files. If None, the function is determined
        by the file extension. Refer to :ref:`gaze-io` for available function names. (default: None)
    load_kwargs: dict[str, Any] | None
        A dictionary of additional keyword arguments that are passed to the ``load_function``.
        (default: None)
    """

    content: str

    filename: str | None = None
    url: str | None = None
    mirrors: list[str] | None = None
    md5: str | None = None

    filename_pattern: str | None = None
    filename_pattern_schema_overrides: dict[str, type] | None = None

    load_function: str | None = None
    load_kwargs: dict[str, Any] | None = None

    @staticmethod
    def from_dict(dictionary: dict[str, Any]) -> ResourceDefinition:
        """Create a ``Resource`` instance from a dictionary.

        Parameters
        ----------
        dictionary : dict[str, Any]
            A dictionary containing Resource parameters.

        Returns
        -------
        ResourceDefinition
            An initialized ``Resource`` instance.
        """
        if 'resource' in dictionary:
            warn(
                DeprecationWarning(
                    'from_dict() key "resource" is deprecated since version v0.23.0. '
                    'Please use key "url" instead. '
                    'This field will be removed in v0.28.0.',
                ),
            )

            url = dictionary['resource']
            dictionary = {key: value for key, value in dictionary.items() if key != 'resource'}
            dictionary['url'] = url

        return ResourceDefinition(**dictionary)

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """Convert the ``ResourceDefinition`` instance into a dictionary.

        Parameters
        ----------
        exclude_none: bool
            Exclude attributes that are either ``None`` or that are objects that evaluate to
            ``False`` (e.g., ``[]``, ``{}``). Attributes of type ``bool``, ``int``, and ``float``
            are not excluded.

        Returns
        -------
        dict[str, Any]
            ``dict`` representation of ``ResourceDefinition``.
        """
        data = asdict(self)

        # Exclude fields that evaluate to False (False, None, [], {})
        if exclude_none:
            for key, value in list(data.items()):
                if not isinstance(value, (bool, int, float)) and not value:
                    del data[key]

        return data


class ResourceDefinitions(list):
    """List of :py:class:`~pymovements.ResourceDefinition` instances.

    Parameters
    ----------
    resources: Iterable[ResourceDefinition | dict[str, Any]] | None
        An iterable of :py:class:`~.ResourceDefinition` instances or dictionaries containing
        :py:class:`~.ResourceDefinition` parameters. In case an element is a dictionary it will be
        converted using :py:meth:`~.ResourceDefinition.from_dict()`.
    """

    def __init__(
            self, resources: Iterable[ResourceDefinition | dict[str, Any]] | None = None,
    ) -> None:
        if resources is None:
            resources = []

        _resources: Iterable[ResourceDefinition] = [
            resource if isinstance(resource, ResourceDefinition)
            else ResourceDefinition.from_dict(resource)
            for resource in resources
        ]
        super().__init__(_resources)

    def filter(self, content: str | None = None) -> ResourceDefinitions:
        """Filter ``ResourceDefinitions`` for content type.

        Parameters
        ----------
        content: str | None
            The content type to filter for. If ``None``, then don't filter. (default: None)

        Returns
        -------
        ResourceDefinitions
            A new ``ResourceDefinitions`` instance that contains only resources of the specified
            content type.
        """
        if content is None:
            return self

        resources = [resource for resource in self if resource.content == content]
        return ResourceDefinitions(resources)

    @staticmethod
    @deprecated(
        reason='Please use ResourceDefinitions.from_dicts() instead. '
               'This property will be removed in v0.28.0.',
        version='v0.23.0',
    )
    def from_dict(
        dictionary: dict[str, Sequence[dict[str, Any]]] | None,
    ) -> ResourceDefinitions:
        """Create a ``ResourceDefinitions`` instance from a dictionary of lists of dictionaries.

        Parameters
        ----------
        dictionary : dict[str, Sequence[dict[str, Any]]] | None
            A list of dictionaries containing ``ResourceDefinition`` parameters.

        Returns
        -------
        ResourceDefinitions
            An initialized ``ResourceDefinitions`` instance.
        """
        if dictionary is None:
            return ResourceDefinitions()

        resources = []
        for content_type, content_dictionaries in dictionary.items():
            if not content_dictionaries:
                continue
            for content_dictionary in content_dictionaries:
                _dictionary = deepcopy(content_dictionary)
                _dictionary['content'] = content_type
                resource = ResourceDefinition.from_dict(_dictionary)
                resources.append(resource)

        return ResourceDefinitions(resources)

    @staticmethod
    def from_dicts(dictionaries: Sequence[dict[str, Any]] | None) -> ResourceDefinitions:
        """Create a ``ResourceDefinitions`` instance from a list of dictionaries.

        Parameters
        ----------
        dictionaries : Sequence[dict[str, Any]] | None
            A list of dictionaries containing ``ResourceDefinition`` parameters.

        Returns
        -------
        ResourceDefinitions
            An initialized ``ResourceDefinitions`` instance.
        """
        if dictionaries is None:
            return ResourceDefinitions()

        resources = [ResourceDefinition.from_dict(dictionary) for dictionary in dictionaries]

        return ResourceDefinitions(resources)

    def to_dicts(self, *, exclude_none: bool = True) -> list[dict[str, Any]]:
        """Convert the ``ResourceDefinitions`` instance into a list of dictionaries.

        Parameters
        ----------
        exclude_none: bool
            Exclude attributes that are either ``None`` or that are objects that evaluate to
            ``False`` (e.g., ``[]``, ``{}``). Attributes of type ``bool``, ``int``, and ``float``
            are not excluded.

        Returns
        -------
        list[dict[str, Any]]
            ``ResourceDefinition`` as a list of dictionaries.
        """
        return [resource.to_dict(exclude_none=exclude_none) for resource in self]

    def has_content(self, content: str) -> bool:
        """Check if any ``ResourceDefinition`` has specific content.

        Parameters
        ----------
        content: str
            content type

        Returns
        -------
        bool
            ``True`` if contains ``ResourceDefinition`` of specific content type.
        """
        return any(resource.content == content for resource in self)


class _HasResourcesIndexer:
    """Helper class for :py:meth:`~pymovements.dataset.DatasetDefinition.has_resources` property.

    Provides dynamic inference on the presence of any
    :py:meth:`~pymovements.dataset.DatasetDefinition.resources`.
    """

    def __init__(self, resources: ResourceDefinitions) -> None:
        self._resources = resources

    def set_resources(self, resources: ResourceDefinitions) -> None:
        """Set dataset definition resources for lookup."""
        self._resources = resources

    def __getitem__(self, key: str) -> bool:
        """Lookup if resources of specific content are set."""
        return self.__bool__() and self._resources.has_content(key)

    def __bool__(self) -> bool:
        """Lookup if resources of any content are set."""
        return bool(self._resources)

    def __eq__(self, other: Any) -> bool:
        """Return self == other.

        Automatically casts to bool if compared to a boolean.
        """
        if isinstance(other, bool):  # Needed to check equality against booleans.
            return self.__bool__() == other
        return super().__eq__(other)

    def __repr__(self) -> str:
        """Return string with boolean value wheter any resources are set."""
        return str(self.__bool__())

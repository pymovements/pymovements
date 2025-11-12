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
"""DatasetPaths module."""
from __future__ import annotations

from pathlib import Path

from pymovements._utils._html import repr_html


@repr_html()
class DatasetPaths:
    """Defines the paths of a dataset.

    Parameters
    ----------
    root: str | Path
        Path to the root directory of the dataset. (default: 'data')
    dataset: str | None
        Dataset directory name under root path. Can be `.` if dataset is located in root path.
        (default: None)
    raw: str
        Name of directory under dataset path that contains raw data. Can be `.` if raw data is
        located in dataset path. We advise the user to keep the original raw data separate from
        the preprocessed / event data. (default: 'raw')
    events: str
        Name of directory under dataset path that will be used to store event data. We advise
        the user to keep the event data separate from the original raw data. (default: 'events')
    precomputed_events: str
        Name of directory under dataset path that contains precomputed event data.
        Can be `.` if precomputed event data is located in dataset path.
        We advise the user to keep the original precomputed event data separate
        from the preprocessed / event data. (default: 'precomputed_events')
    precomputed_reading_measures: str
        Name of directory under dataset path that contains precomputed reading measure data.
        Can be `.` if precomputed event data is located in dataset path.
        We advise the user to keep the original precomputed event data separate
        from the preprocessed / event data. (default: 'precomputed_reading_measures')
    preprocessed: str
        Name of directory under dataset path that will be used to store preprocessed data. We
        advise the user to keep the preprocessed data separate from the original raw data.
        (default: 'preprocessed')
    downloads: str
        Name of directory to store downloaded data. (default: 'downloads')
    """

    def __init__(
            self,
            *,
            root: str | Path = 'data',
            dataset: str | None = None,
            raw: str = 'raw',
            events: str = 'events',
            precomputed_events: str = 'precomputed_events',
            precomputed_reading_measures: str = 'precomputed_reading_measures',
            preprocessed: str = 'preprocessed',
            downloads: str = 'downloads',
    ):
        self._root = Path(root)
        self._dataset = dataset
        self._raw = raw
        self._events = events
        self._precomputed_events = precomputed_events
        self._precomputed_reading_measures = precomputed_reading_measures
        self._preprocessed = preprocessed
        self._downloads = downloads

    def get_preprocessed_filepath(
            self,
            raw_filepath: Path,
            *,
            preprocessed_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Path:
        """Get preprocessed filepath in accordance to filepath of the raw file.

        The preprocessed filepath will point to a feather file.

        Parameters
        ----------
        raw_filepath: Path
            The Path to the raw file.
        preprocessed_dirname: str | None
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.preprocessed_rootpath`. (default: None)
        extension: str
            extension specifies the fileformat to store the data. (default: 'feather')

        Returns
        -------
        Path
            The Path to the preprocessed feather file.
        """
        relative_raw_dirpath = raw_filepath.parent
        relative_raw_dirpath = relative_raw_dirpath.relative_to(self.raw)

        if preprocessed_dirname is None:
            preprocessed_rootpath = self.preprocessed
        else:
            preprocessed_rootpath = self.dataset / preprocessed_dirname

        preprocessed_file_dirpath = preprocessed_rootpath / relative_raw_dirpath

        # Get new filename for saved feather file.
        preprocessed_filename = raw_filepath.stem + '.' + extension

        return preprocessed_file_dirpath / preprocessed_filename

    def raw_to_event_filepath(
            self,
            raw_filepath: Path,
            *,
            events_dirname: str | None = None,
            extension: str = 'feather',
    ) -> Path:
        """Get event filepath in accordance to filepath of the raw file.

        The event filepath will point to file with the specified extension.

        Parameters
        ----------
        raw_filepath: Path
            The Path to the raw file.
        events_dirname: str | None
            One-time usage of an alternative directory name to save data relative to
            :py:meth:`pymovements.Dataset.path`.
            This argument is used only for this single call and does not alter
            :py:meth:`pymovements.Dataset.events_rootpath`.
            (default: None)
        extension: str
            extension specifies the fileformat to store the data. (default: 'feather')

        Returns
        -------
        Path
            The Path to the event feather file.
        """
        relative_raw_dirpath = raw_filepath.parent
        relative_raw_dirpath = relative_raw_dirpath.relative_to(self.raw)

        if events_dirname is None:
            events_rootpath = self.events
        else:
            events_rootpath = self.dataset / events_dirname

        events_file_dirpath = events_rootpath / relative_raw_dirpath

        # Get new filename for saved feather file.
        events_filename = raw_filepath.stem + '.' + extension

        return events_file_dirpath / events_filename

    @property
    def root(self) -> Path:
        """The root path to your dataset.

        Returns
        -------
        Path
            The root path to your dataset.

        Example
        -------
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset')
        >>> dataset.paths.root# doctest: +SKIP
        Path('/path/to/your/dataset')

        This is the same as:
        >>> paths = pm.DatasetPaths(root='/path/to/your/dataset', dataset='.')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.root# doctest: +SKIP
        Path('/path/to/your/dataset')

        The root stays unaffected by the dataset directory name:
        >>> paths = pm.DatasetPaths(root='/path/to/your/dataset', dataset='your_dataset')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.root# doctest: +SKIP
        Path('/path/to/your/dataset')
        """
        return self._root

    @property
    def dataset(self) -> Path:
        """The path to the dataset directory.

        Returns
        -------
        Path
            The path to the dataset directory.

        Example
        -------
        By passing a `str` or a `Path` as `path` during initialization you can explicitly set the
        directory path of the dataset:
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset')
        >>> dataset.path# doctest: +SKIP
        Path('/path/to/your/dataset')

        If you just want to specify the root directory path which holds all your local datasets, you
        can create pass a :py:class:`~pymovements.dataset.DatasetPaths` object and set the `root`:
        >>> paths = pm.DatasetPaths(root='/path/to/your/common/root/')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.path# doctest: +SKIP
        Path('/path/to/your/common/root/ToyDataset')

        You can also specify an alternative dataset directory name:
        >>> paths = pm.DatasetPaths(root='/path/to/your/common/root/', dataset='my_dataset')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.path# doctest: +SKIP
        Path('/path/to/your/common/root/my_dataset')

        If your dataset is not in a separate directory under the root path then you can also
        specify `.` as the directory name. We discourage this and advise the user to have a
        directory which holds all datasets with sub-directories using the registered dataset names.
        ... dataset = Dataset(
        >>> paths = pm.DatasetPaths(root='/path/to/your/dataset/', dataset='.')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.path# doctest: +SKIP
        Path('/path/to/your/dataset')
        """
        if self._dataset is None:
            return self._root
        return self._root / self._dataset

    def fill_name(self, name: str) -> None:
        """Fill dataset directory name with dataset name.

        Dataset directory name will only be updated if :py:attr:pymovements.DatasetPaths.dataset` is
        ``None``.

        Parameters
        ----------
        name: str
            Name to update dataset directory name with. The dataset directory name will only be
            updated if :py:attr:pymovements.DatasetPaths.dataset` is ``None``.
        """
        if self._dataset is None:
            self._dataset = name

    @property
    def events(self) -> Path:
        """The path to the directory of the event data.

        The path points to the events directory under the dataset path.

        Returns
        -------
        Path
            The path to the events directory.

        Example
        -------
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset/')
        >>> dataset.paths.events# doctest: +SKIP
        Path('/path/to/your/dataset/events')

        If you just want to specify the root directory path which holds all your local datasets, you
        can create pass a :py:class:`~pymovements.dataset.DatasetPaths` object and set the `root`:
        >>> paths = pm.DatasetPaths(root='/path/to/your/common/root/')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.events# doctest: +SKIP
        Path('/path/to/your/common/root/ToyDataset/events')

        This way you can also explicitely specify the events directory name. The default is
        `events`.
        >>> paths = pm.DatasetPaths(root='/path/to/your/datasets/', events='my_events')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.events# doctest: +SKIP
        Path('/path/to/your/datasets/ToyDataset/my_events')
        """
        return self.dataset / self._events

    @property
    def preprocessed(self) -> Path:
        """The path to the directory of the preprocessed gaze data.

        The path points to the preprocessed data directory under the dataset path.

        Returns
        -------
        Path
            The path to the preprocessed data directory.

        Example
        -------
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset/')
        >>> dataset.paths.preprocessed# doctest: +SKIP
        Path('/path/to/your/dataset/preprocessed')

        If you just want to specify the root directory path which holds all your local datasets, you
        can create pass a :py:class:`~pymovements.dataset.DatasetPaths` object and set the `root`:
        >>> paths = pm.DatasetPaths(root='path/to/your/common/root/')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.preprocessed# doctest: +SKIP
        Path('path/to/your/common/root/ToyDataset/preprocessed')

        This way you can also explicitely specify the events directory name. The default is
        `preprocessed`.
        >>> paths = pm.DatasetPaths(
        ...     root='/path/to/your/datasets/', preprocessed='my_preprocessed_data',
        ... )
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.preprocessed# doctest: +SKIP
        Path('/path/to/your/datasets/ToyDataset/my_preprocessed_data')
        """
        return self.dataset / self._preprocessed

    @property
    def raw(self) -> Path:
        """The path to the directory of the raw data.

        The path points to the raw data directory under the dataset path.

        Returns
        -------
        Path
            Path to the raw data directory.

        Example
        -------
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset/')
        >>> dataset.paths.raw# doctest: +SKIP
        Path('/path/to/your/dataset/raw')

        If you just want to specify the root directory path which holds all your local datasets, you
        can create pass a :py:class:`~pymovements.dataset.DatasetPaths` object and set the `root`:
        >>> paths = pm.DatasetPaths(root='path/to/your/common/root/')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.raw# doctest: +SKIP
        Path('path/to/your/common/root/ToyDataset/raw')

        This way you can also explicitely specify the raw directory name. The default is `raw`.
        >>> paths = pm.DatasetPaths(root='/path/to/your/datasets/', raw='my_raw')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.raw# doctest: +SKIP
        Path('/path/to/your/datasets/ToyDataset/my_raw')

        If your raw data is not in a separate directory under the root path then you can also
        specify `.` as the directory name. We discourage this and advise the user to keep raw data
        and preprocessed data separated.
        >>> paths = pm.DatasetPaths(root='/path/to/your/datasets/', raw='.')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.raw# doctest: +SKIP
        Path('/path/to/your/datasets/ToyDataset')
        """
        return self.dataset / self._raw

    @property
    def precomputed_events(self) -> Path:
        """The path to the directory of the precomputed event data.

        The path points to the precomputed event data directory under the dataset path.

        Returns
        -------
        Path
            Path to the precomputed event data directory.

        Example
        -------
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset/')
        >>> dataset.paths.precomputed_events# doctest: +SKIP
        Path('/path/to/your/dataset/precomputed_events')

        If you just want to specify the root directory path which holds all your local datasets, you
        can create pass a :py:class:`~pymovements.dataset.DatasetPaths` object and set the `root`:
        >>> paths = pm.DatasetPaths(root='path/to/your/common/root/')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.precomputed_events# doctest: +SKIP
        Path('path/to/your/common/root/ToyDataset/precomputed_events')

        This way you can also explicitely specify the precomputed directory name.
        The default is `precomputed_events`.
        >>> paths = pm.DatasetPaths(root='/path/to/your/datasets/', precomputed_events='my_pe')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.precomputed_events# doctest: +SKIP
        Path('/path/to/your/datasets/ToyDataset/my_pe')

        If your precomputed event data is not in a separate directory under the root path then you
        can also specify `.` as the directory name. We discourage this and advise the user to keep
        precomputed and preprocessed data separated.
        >>> paths = pm.DatasetPaths(root='/path/to/your/datasets/', precomputed_events='.')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.precomputed_events# doctest: +SKIP
        Path('/path/to/your/datasets/ToyDataset')
        """
        return self.dataset / self._precomputed_events

    @property
    def precomputed_reading_measures(self) -> Path:
        """The path to the directory of the precomputed reading measures.

        The path points to the precomputed reading measure data directory under the dataset path.

        Returns
        -------
        Path
            Path to the precomputed reading measure data directory.

        Example
        -------
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset/')
        >>> dataset.paths.precomputed_reading_measures# doctest: +SKIP
        Path('/path/to/your/dataset/precomputed_reading_measures')

        If you just want to specify the root directory path which holds all your local datasets, you
        can create pass a :py:class:`~pymovements.dataset.DatasetPaths` object and set the `root`:
        >>> paths = pm.DatasetPaths(root='path/to/your/common/root/')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.precomputed_reading_measures# doctest: +SKIP
        Path('path/to/your/common/root/ToyDataset/precomputed_reading_measures')

        This way you can also explicitely specify the raw directory name. The default is
        `precomputed_rm`.
        >>> paths = pm.DatasetPaths(root='/path/to/your/datasets/', raw='my_precomputed_rm')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.precomputed_reading_measures# doctest: +SKIP
        Path('/path/to/your/datasets/ToyDataset/my_precomputed_rm')

        If your precomputed event  data is not in a separate directory under the root path then you
        can also specify `.` as the directory name. We discourage this and advise the user to keep
        precomputed data and preprocessed data separated.
        >>> paths = pm.DatasetPaths(root='/path/to/your/datasets/',precomputed_reading_measures='.')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.precomputed_events# doctest: +SKIP
        Path('/path/to/your/datasets/ToyDataset')
        """
        return self.dataset / self._precomputed_reading_measures

    @property
    def downloads(self) -> Path:
        """The path to the directory of the raw data.

        The download path points to the download directory under the root path.

        Returns
        -------
        Path
            The path to the download directory under the root path.

        Example
        -------
        >>> import pymovements as pm
        >>>
        >>> dataset = pm.Dataset("ToyDataset", path='/path/to/your/dataset/')
        >>> dataset.paths.downloads# doctest: +SKIP
        Path('/path/to/your/dataset/downloads')

        If you just want to specify the root directory path which holds all your local datasets, you
        can create pass a :py:class:`~pymovements.dataset.DatasetPaths` object and set the `root`:
        >>> paths = pm.DatasetPaths(root='path/to/your/common/root/')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.downloads# doctest: +SKIP
        Path('path/to/your/common/root/ToyDataset/downloads')

        This way you can also explicitely specify the download directory name. The default is
        `downloads`.
        >>> paths = pm.DatasetPaths(root='/path/to/your/datasets/', downloads='my_downloads')
        >>> dataset = pm.Dataset("ToyDataset", path=paths)
        >>> dataset.paths.downloads# doctest: +SKIP
        Path('/path/to/your/datasets/ToyDataset/my_downloads')
        """
        return self.dataset / self._downloads

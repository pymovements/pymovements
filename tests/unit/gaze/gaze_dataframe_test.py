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
"""Tests deprecated GazeDataFrame alias for Gaze."""
import pytest

from pymovements import __version__
from pymovements import Gaze
from pymovements import GazeDataFrame


@pytest.fixture(name='gaze_subclass')
def fixture_gaze_subclass():
    class GazeSubclass(Gaze):
        ...
    yield GazeSubclass


@pytest.fixture(name='gaze_df_subclass')
def fixture_gaze_df_subclass():
    class GazeDataFrameSubclass(GazeDataFrame):
        ...
    yield GazeDataFrameSubclass


@pytest.fixture(name='gaze_df_subsubclass')
def fixture_gaze_df_subsubclass(gaze_df_subclass):
    class GazeDataFrameSubSubclass(gaze_df_subclass):
        ...

    yield GazeDataFrameSubSubclass


def test_gaze_issubclass_gaze_df():
    assert issubclass(Gaze, GazeDataFrame)


def test_gaze_df_issubclass_gaze_df():
    assert issubclass(GazeDataFrame, GazeDataFrame)


def test_gaze_subclass_issubclass_gaze_df(gaze_subclass):
    assert issubclass(gaze_subclass, GazeDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_df_subclass_issubclass_gaze(gaze_df_subclass):
    assert issubclass(gaze_df_subclass, Gaze)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_df_subclass_issubclass_gazedf(gaze_df_subclass):
    assert issubclass(gaze_df_subclass, GazeDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_df_subsubclass_issubclass_gaze(gaze_df_subsubclass):
    assert issubclass(gaze_df_subsubclass, Gaze)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_df_subsubclass_issubclass_gaze_df(gaze_df_subsubclass):
    assert issubclass(gaze_df_subsubclass, GazeDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_df_isinstance_gaze_df():
    gaze_df = GazeDataFrame()
    assert isinstance(gaze_df, GazeDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_df_subclass_isinstance_gaze_df(gaze_df_subclass):
    gaze_df = gaze_df_subclass()
    assert isinstance(gaze_df, GazeDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_df_subsubclass_isinstance_gaze_df(gaze_df_subsubclass):
    gaze_df = gaze_df_subsubclass()
    assert isinstance(gaze_df, GazeDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_df_subsubclass_isinstance_gaze(gaze_df_subsubclass):
    gaze_df = gaze_df_subsubclass()
    assert isinstance(gaze_df, Gaze)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_df_isinstance_gaze():
    gaze_df = GazeDataFrame()
    assert isinstance(gaze_df, Gaze)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_df_subclass_isinstance_gaze(gaze_df_subclass):
    gaze_df = gaze_df_subclass()
    assert isinstance(gaze_df, Gaze)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_isinstance_gaze_df():
    gaze = Gaze()
    assert isinstance(gaze, GazeDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_gaze_subclass_isinstance_gaze_df(gaze_subclass):
    gaze = gaze_subclass()
    assert isinstance(gaze, GazeDataFrame)


def test_is_gaze_df_deprecated():
    with pytest.raises(DeprecationWarning):
        GazeDataFrame()


def test_is_gaze_df_subclass_deprecated():
    # pylint: disable=unused-variable
    with pytest.raises(DeprecationWarning):
        class AnotherGazeDataFrameSubclass(GazeDataFrame):
            ...


def test_is_gaze_df_dubplicate_subclass_deprecated():
    # pylint: disable=unused-variable
    with pytest.raises(DeprecationWarning):
        class AnotherGazeDataFrameSubclass(GazeDataFrame):
            ...


def test_is_gaze_df_subsubclass_deprecated():
    # pylint: disable=unused-variable
    with pytest.raises(DeprecationWarning):
        class YetAnotherGazeDataFrameSubclass(GazeDataFrame):
            ...

        with pytest.raises(DeprecationWarning):
            class AnotherGazeDataFrameSubSubclass(YetAnotherGazeDataFrameSubclass):
                ...


def test_is_gaze_df_removed(assert_deprecation_is_removed):
    with pytest.raises(DeprecationWarning) as info:
        GazeDataFrame()
    assert_deprecation_is_removed('GazeDataFrame', info.value.args[0], __version__)

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
"""Tests deprecated EventDataFrame alias for Events."""
import pytest

from pymovements import __version__
from pymovements import EventDataFrame
from pymovements import Events


@pytest.fixture(name='events_subclass')
def fixture_events_subclass():
    class EventsSubclass(Events):
        ...
    yield EventsSubclass


@pytest.fixture(name='event_df_subclass')
def fixture_event_df_subclass():
    class EventDataFrameSubclass(EventDataFrame):
        ...
    yield EventDataFrameSubclass


@pytest.fixture(name='event_df_subsubclass')
def fixture_event_df_subsubclass(event_df_subclass):
    class EventDataFrameSubSubclass(event_df_subclass):
        ...

    yield EventDataFrameSubSubclass


def test_events_issubclass_event_df():
    assert issubclass(Events, EventDataFrame)


def test_event_df_issubclass_event_df():
    assert issubclass(EventDataFrame, EventDataFrame)


def test_events_subclass_issubclass_event_df(events_subclass):
    assert issubclass(events_subclass, EventDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_event_df_subclass_issubclass_events(event_df_subclass):
    assert issubclass(event_df_subclass, Events)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_event_df_subclass_issubclass_eventsdf(event_df_subclass):
    assert issubclass(event_df_subclass, EventDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_event_df_subsubclass_issubclass_events(event_df_subsubclass):
    assert issubclass(event_df_subsubclass, Events)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_event_df_subsubclass_issubclass_event_df(event_df_subsubclass):
    assert issubclass(event_df_subsubclass, EventDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_event_df_isinstance_event_df():
    event_df = EventDataFrame()
    assert isinstance(event_df, EventDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_event_df_subclass_isinstance_event_df(event_df_subclass):
    event_df = event_df_subclass()
    assert isinstance(event_df, EventDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_event_df_subsubclass_isinstance_event_df(event_df_subsubclass):
    event_df = event_df_subsubclass()
    assert isinstance(event_df, EventDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_event_df_subsubclass_isinstance_events(event_df_subsubclass):
    event_df = event_df_subsubclass()
    assert isinstance(event_df, Events)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_event_df_isinstance_events():
    event_df = EventDataFrame()
    assert isinstance(event_df, Events)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_event_df_subclass_isinstance_events(event_df_subclass):
    event_df = event_df_subclass()
    assert isinstance(event_df, Events)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_events_isinstance_event_df():
    events = Events()
    assert isinstance(events, EventDataFrame)


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_events_subclass_isinstance_event_df(events_subclass):
    events = events_subclass()
    assert isinstance(events, EventDataFrame)


def test_is_event_df_deprecated():
    with pytest.raises(DeprecationWarning):
        EventDataFrame()


def test_is_event_df_subclass_deprecated():
    # pylint: disable=unused-variable
    with pytest.raises(DeprecationWarning):
        class AnotherEventDataFrameSubclass(EventDataFrame):
            ...


def test_is_event_df_dubplicate_subclass_deprecated():
    # pylint: disable=unused-variable
    with pytest.raises(DeprecationWarning):
        class AnotherEventDataFrameSubclass(EventDataFrame):
            ...


def test_is_event_df_subsubclass_deprecated():
    # pylint: disable=unused-variable
    with pytest.raises(DeprecationWarning):
        class YetAnotherEventDataFrameSubclass(EventDataFrame):
            ...

        with pytest.raises(DeprecationWarning):
            class AnotherEventDataFrameSubSubclass(YetAnotherEventDataFrameSubclass):
                ...


def test_is_event_df_removed(assert_deprecation_is_removed):
    with pytest.raises(DeprecationWarning) as info:
        EventDataFrame()
    assert_deprecation_is_removed('EventDatFrame', info.value.args[0], __version__)

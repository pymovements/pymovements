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
"""Test metadata attribute of the Events class."""
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm
from pymovements.events import Events
from pymovements.stimulus.text import TextStimulus


def test_events_metadata_defaults_to_empty_dict():
    events = Events()
    assert events.metadata == {}


def test_events_metadata_is_stored():
    events = Events(metadata={'Sources': ['raw/sub_1.csv'], 'subject_id': 1})
    assert events.metadata == {'Sources': ['raw/sub_1.csv'], 'subject_id': 1}


def test_events_metadata_is_deepcopied_on_construction():
    metadata = {'Sources': ['raw/sub_1.csv'], 'offset': {'inclusive': True}}

    events = Events(onsets=[0], offsets=[1], sampling_rate=1000, metadata=metadata)

    # Neither side reaches the other, nested values included.
    events.metadata['Sources'].append('other.csv')
    metadata['offset']['inclusive'] = False
    assert metadata == {'Sources': ['raw/sub_1.csv'], 'offset': {'inclusive': False}}
    assert events.metadata['offset'] == {'inclusive': True}


def test_events_clone_deepcopies_metadata():
    events = Events(
        name=['fixation'], onsets=[0], offsets=[1],
        metadata={'Sources': ['raw/sub_1.csv']},
    )

    events_clone = events.clone()

    assert events_clone.metadata == events.metadata
    events_clone.metadata['Sources'].append('other.csv')
    assert events.metadata['Sources'] == ['raw/sub_1.csv']


def test_events_split_list_propagates_metadata():
    events = Events(
        name=['fixation', 'fixation'], onsets=[0, 2], offsets=[1, 3], trials=[1, 2],
        metadata={'Sources': ['raw/sub_1.csv']},
    )

    splits = events.split(by='trial')

    assert len(splits) == 2
    for events_split in splits:
        assert events_split.metadata == {'Sources': ['raw/sub_1.csv']}


def test_events_split_dict_propagates_metadata():
    events = Events(
        name=['fixation', 'fixation'], onsets=[0, 2], offsets=[1, 3], trials=[1, 2],
        metadata={'Sources': ['raw/sub_1.csv']},
    )

    splits = events.split(by='trial', as_dict=True)

    assert len(splits) == 2
    for events_split in splits.values():
        assert events_split.metadata == {'Sources': ['raw/sub_1.csv']}


def make_correction_events():
    return Events(
        pl.DataFrame({
            'name': ['fixation'],
            'onset': [0],
            'offset': [100],
            'location': [[100.0, 105.0]],
        }),
        metadata={'Sources': ['raw/sub_1.csv']},
    )


def make_correction_stimulus():
    return TextStimulus(
        aois=pl.DataFrame({
            'word': ['first'],
            'start_x': [90.0],
            'start_y': [80.0],
            'end_x': [200.0],
            'end_y': [120.0],
        }),
        aoi_column='word',
        start_x_column='start_x',
        start_y_column='start_y',
        end_x_column='end_x',
        end_y_column='end_y',
        metadata={'Sources': ['stimuli/text_1.csv']},
    )


def test_events_correct_fixations_inplace_merges_stimulus_sources():
    events = make_correction_events()
    stimulus = make_correction_stimulus()

    events.correct_fixations(stimulus, algorithm='attach')

    assert events.metadata['Sources'] == ['raw/sub_1.csv', 'stimuli/text_1.csv']


def test_events_correct_fixations_not_inplace_propagates_metadata():
    events = make_correction_events()
    stimulus = make_correction_stimulus()

    corrected = events.correct_fixations(stimulus, algorithm='attach', inplace=False)

    # The new object unions the event sources with the stimulus sources.
    assert corrected.metadata == {'Sources': ['raw/sub_1.csv', 'stimuli/text_1.csv']}
    # The original events and their metadata stay untouched and unshared.
    corrected.metadata['Sources'].append('other.csv')
    assert events.metadata == {'Sources': ['raw/sub_1.csv']}


def test_events_map_to_aois_merges_stimulus_sources(simple_stimulus: TextStimulus) -> None:
    simple_stimulus.metadata['Sources'] = ['stimuli/text_1_aoi.csv']

    events = Events(
        data=pl.DataFrame({
            'name': ['fixation'],
            'onset': [0],
            'offset': [1],
            'location': [[5, 5]],
        }),
        metadata={'Sources': ['raw/sub_1.csv']},
    )

    events.map_to_aois(simple_stimulus)

    assert events.metadata['Sources'] == ['raw/sub_1.csv', 'stimuli/text_1_aoi.csv']


_REBUILD_METADATA = {'key': 'value', 'offset': {'inclusive': False}, 'sampling_rate': 1000.0}
_REBUILD_ENTRY_COLUMNS = ['trial', 'onset', 'duration', 'name', 'offset', 'location']


def make_entry_events():
    return Events(
        pl.DataFrame({
            'trial': [1, 1, 2],
            'name': ['fixation', 'fixation', 'fixation'],
            'onset': [0, 200, 400],
            'duration': [100, None, 100],
            'offset': [100, 300, 500],
            'location': [[100.0, 105.0], [110.0, 195.0], [120.0, 302.0]],
        }),
        trial_columns='trial',
        offsets_inclusive=False,
        sampling_rate=1000,
        metadata={'key': 'value'},
    )


@pytest.fixture(name='entry_events')
def fixture_entry_events():
    # The construction warns about the null duration, see the test below. The rebuild sites
    # construct with validate=False, so the warning must not repeat there.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*null duration')
        return make_entry_events()


@pytest.fixture(name='entry_stimulus')
def fixture_entry_stimulus():
    return TextStimulus(
        aois=pl.DataFrame({
            'word': ['first', 'second', 'third'],
            'word_idx': [0, 1, 2],
            'start_x': [90.0, 90.0, 90.0],
            'start_y': [80.0, 180.0, 280.0],
            'end_x': [200.0, 200.0, 200.0],
            'end_y': [120.0, 220.0, 320.0],
        }),
        aoi_column='word',
        start_x_column='start_x',
        start_y_column='start_y',
        end_x_column='end_x',
        end_y_column='end_y',
    )


def make_gaze():
    return pm.Gaze(
        samples=pl.DataFrame({
            'time': list(range(0, 600)) * 2,
            'trial': [1] * 600 + [2] * 600,
            'x': [0.0] * 1200,
            'y': [0.0] * 1200,
        }),
        time_column='time',
        pixel_columns=['x', 'y'],
        trial_columns='trial',
    )


@contextmanager
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # The detector itself may report that it detected nothing.
        warnings.filterwarnings('ignore', message='.*No events were detected')
        yield


def assert_rebuilt(rebuilt, expected_columns):
    assert rebuilt
    for events in rebuilt:
        assert events.frame.columns == expected_columns
        assert events.metadata.items() >= _REBUILD_METADATA.items()


def test_entry_events_construction_warns_once_about_the_null_duration():
    with pytest.warns(UserWarning, match='1 row has a null duration') as record:
        make_entry_events()

    assert len(record) == 1


def test_clone_rebuilds_entries_without_warning(entry_events):
    with no_warnings():
        cloned = entry_events.clone()

    assert_rebuilt([cloned], _REBUILD_ENTRY_COLUMNS)
    assert_frame_equal(cloned.frame, entry_events.frame)


def test_split_rebuilds_entries_without_warning(entry_events):
    with no_warnings():
        splits = entry_events.split()

    assert_rebuilt(splits, _REBUILD_ENTRY_COLUMNS)


def test_split_dict_rebuilds_entries_without_warning(entry_events):
    with no_warnings():
        splits = entry_events.split(as_dict=True)

    assert_rebuilt(list(splits.values()), _REBUILD_ENTRY_COLUMNS)


def test_correct_fixations_rebuilds_entries_without_warning(entry_events, entry_stimulus):
    with no_warnings():
        corrected = entry_events.correct_fixations(
            entry_stimulus, algorithm='attach', inplace=False,
        )

    assert_rebuilt(
        [corrected], [*_REBUILD_ENTRY_COLUMNS, 'location_original', 'correction_algorithm'],
    )


def test_gaze_detect_rebuilds_shared_empty_events_without_warning():
    gaze = make_gaze()
    # A convention entry without its column: a validating construction would warn about the
    # orphan entry, the rebuild must not.
    gaze.events.metadata = deepcopy(_REBUILD_METADATA)

    with no_warnings():
        gaze.detect('fill')

    # The fill detector appends its offsets to the empty events.
    assert_rebuilt([gaze.events], ['trial', 'onset', 'duration', 'name', 'offset'])


def test_gaze_detect_rebuilds_per_group_events_without_warning(entry_events):
    gaze = make_gaze()
    gaze.events = entry_events

    with no_warnings():
        gaze.detect('fill')

    assert_rebuilt([gaze.events], _REBUILD_ENTRY_COLUMNS)


def test_bridge_gaps_rebuilds_entries_without_warning(entry_events):
    with no_warnings():
        bridged = pm.synthetic.bridge_gaps(entry_events)

    assert_rebuilt([bridged], _REBUILD_ENTRY_COLUMNS)


# measure_reading maps the rebuilt events to the stimulus and discards them, only the measures
# reach the caller, so there are no rebuilt events to check.
def test_measure_reading_does_not_warn(entry_events, entry_stimulus):
    with no_warnings():
        entry_events.measure_reading(entry_stimulus)


def test_measure_reading_dict_does_not_warn(entry_events, entry_stimulus):
    with no_warnings():
        entry_events.measure_reading(
            {1: entry_stimulus, 2: entry_stimulus}, group_columns=['trial'],
        )


@pytest.mark.parametrize('sampling_rate', [1000, 1000.0], ids=['int', 'float'])
def test_metadata_entries_written_by_keywords(sampling_rate):
    events = Events(
        onsets=[0], offsets=[120], offsets_inclusive=True, sampling_rate=sampling_rate,
    )

    assert events.metadata == {'offset': {'inclusive': True}, 'sampling_rate': sampling_rate}
    assert type(events.metadata['sampling_rate']) is type(sampling_rate)


@pytest.mark.parametrize(
    ('offsets_inclusive', 'expected_ms'),
    [
        pytest.param(True, [121, 38], id='inclusive'),
        pytest.param(False, [120, 37], id='exclusive'),
    ],
)
def test_metadata_entries_declare_and_rate_rebuild_without_keywords(
        offsets_inclusive, expected_ms,
):
    events = Events(
        onsets=[0, 121], offsets=[120, 158],
        offsets_inclusive=offsets_inclusive, sampling_rate=1000,
    )

    rebuilt = Events(events.frame.drop('duration'), metadata=events.metadata)

    assert rebuilt.frame['duration'].to_list() == [timedelta(milliseconds=v) for v in expected_ms]
    assert rebuilt.metadata == {
        'offset': {'inclusive': offsets_inclusive}, 'sampling_rate': 1000,
    }


@pytest.mark.parametrize(
    ('offsets_inclusive', 'shift_ms', 'expected_matches', 'unexpected'),
    [
        pytest.param(
            True, -1,
            [
                r'every row contradicts offsets_inclusive=True from the metadata entry',
                r'durations_from_offsets=True',
                r"metadata\['offset'\] = \{'inclusive': False\} if the offsets are exclusive",
                r'validate=False',
            ],
            ['offsets_inclusive=False'],
            id='inclusive',
        ),
        pytest.param(
            False, 1,
            [
                r'every row contradicts offsets_inclusive=False from the metadata entry',
                r"metadata\['offset'\] = \{'inclusive': True\} if the offsets are inclusive",
                r'validate=False',
            ],
            ['offsets_inclusive=True', 'durations_from_offsets'],
            id='exclusive',
        ),
    ],
)
def test_metadata_entries_rebuild_checks_durations_against_entries(
        offsets_inclusive, shift_ms, expected_matches, unexpected,
):
    events = Events(
        onsets=[0], offsets=[120], offsets_inclusive=offsets_inclusive, sampling_rate=1000,
    )
    frame = events.frame.with_columns(pl.col('duration') + pl.duration(milliseconds=shift_ms))

    with pytest.raises(ValueError) as excinfo:
        Events(frame, metadata=events.metadata)

    message = str(excinfo.value)
    for pattern in expected_matches:
        assert re.search(pattern, message), (pattern, message)
    for substring in unexpected:
        assert substring not in message

    # The offered correction of the entry builds the events without a warning.
    corrected = {**events.metadata, 'offset': {'inclusive': not offsets_inclusive}}
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        rebuilt = Events(frame, metadata=corrected)
    assert rebuilt.metadata == corrected


@pytest.mark.parametrize(
    ('data', 'expected_match'),
    [
        pytest.param(
            pl.DataFrame({'onset': [0], 'offset': [5]}),
            r'offsets_inclusive=False differs from the metadata entry '
            r"metadata\['offset'\] = \{'inclusive': True\}\. "
            r'Correct the entry or drop the keyword\.',
            id='offset',
        ),
        pytest.param(
            pl.DataFrame({'onset': [0], 'duration': [5]}),
            r'offsets_inclusive=False differs from the metadata entry '
            r"metadata\['offset'\] = \{'inclusive': True\}, and the events frame has no "
            r"'offset' column\. Correct the entry or drop the keyword\.",
            id='no_offset',
        ),
    ],
)
def test_metadata_offsets_inclusive_differing_from_entry_raises(data, expected_match):
    with warnings.catch_warnings():
        # The comparison runs before the missing-column warning.
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match=expected_match):
            Events(data, offsets_inclusive=False, metadata={'offset': {'inclusive': True}})


@pytest.mark.parametrize(
    ('sampling_rate', 'expected_match'),
    [
        pytest.param(
            500,
            r'sampling_rate=500 differs from the metadata entry '
            r"metadata\['sampling_rate'\] = 1000\.0\. Correct the entry or drop the keyword\.",
            id='different',
        ),
        pytest.param(
            1000.0004,
            r'sampling_rate=1000\.0004 differs from the metadata entry '
            r"metadata\['sampling_rate'\] = 1000\.0\. Correct the entry or drop the keyword\.",
            id='close',
        ),
    ],
)
@pytest.mark.parametrize(
    'input_kwargs',
    [
        pytest.param({'onsets': [0], 'offsets': [5], 'offsets_inclusive': True}, id='offset'),
        pytest.param({'onsets': [0], 'durations': [5]}, id='no_offset'),
    ],
)
def test_metadata_sampling_rate_differing_from_entry_raises(
        input_kwargs, sampling_rate, expected_match,
):
    with pytest.raises(ValueError, match=expected_match):
        Events(**input_kwargs, sampling_rate=sampling_rate, metadata={'sampling_rate': 1000.0})


def test_metadata_keywords_agreeing_with_entries_pass():
    metadata = {'offset': {'inclusive': True}, 'sampling_rate': 1000.0}

    events = Events(
        onsets=[0], offsets=[5], offsets_inclusive=True, sampling_rate=1000, metadata=metadata,
    )

    assert events.frame['duration'].to_list() == [timedelta(milliseconds=6)]
    assert events.metadata == metadata


def test_metadata_offsets_inclusive_agreeing_with_orphan_entry_warns_for_keyword_and_entry():
    metadata = {'offset': {'inclusive': True}}

    with pytest.warns(UserWarning) as record:
        events = Events(onsets=[0], durations=[5], offsets_inclusive=True, metadata=metadata)

    assert [str(warning.message) for warning in record] == [
        "offsets_inclusive=True ignored: the events frame has no 'offset' column.",
        "metadata holds offset convention entry for the column 'offset', "
        'which the events frame does not have. The metadata is kept.',
    ]
    assert events.metadata == metadata


@pytest.mark.parametrize('sampling_rate', [0, -1])
def test_metadata_non_positive_sampling_rate_entry_raises(sampling_rate):
    with pytest.raises(ValueError, match=r"metadata\['sampling_rate'\] must be positive"):
        Events(onsets=[0], durations=[5], metadata={'sampling_rate': sampling_rate})


@pytest.mark.parametrize(
    ('sampling_rate', 'expected_match'),
    [
        pytest.param('1000', r"must be a number, but is '1000'\.", id='str'),
        pytest.param(True, r'must be a number, but is True\.', id='bool'),
    ],
)
def test_metadata_non_numeric_sampling_rate_entry_raises(sampling_rate, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        Events(onsets=[0], durations=[5], metadata={'sampling_rate': sampling_rate})


@pytest.mark.parametrize('offsets_inclusive', [True, False])
def test_metadata_offsets_inclusive_without_offset_column_writes_no_entry(offsets_inclusive):
    with pytest.warns(UserWarning, match="no 'offset' column"):
        events = Events(onsets=[0], durations=[5], offsets_inclusive=offsets_inclusive)

    assert events.metadata == {}


@pytest.mark.parametrize(
    ('durations', 'offsets'),
    [
        pytest.param([6, 11], [5, 10], id='common_positive_residual'),
        pytest.param([6, 12], [5, 10], id='unequal_residuals'),
        pytest.param(None, [5, 10], id='exclusive_derived'),
    ],
)
def test_metadata_no_sampling_rate_entry_without_sampling_rate(durations, offsets):
    offsets_inclusive = durations is not None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*contradict')
        events = Events(
            onsets=[0, 0], offsets=offsets, durations=durations,
            offsets_inclusive=offsets_inclusive,
        )

    assert 'sampling_rate' not in events.metadata
    assert events.metadata == {'offset': {'inclusive': offsets_inclusive}}


@pytest.mark.parametrize(
    ('metadata', 'expected_match'),
    [
        pytest.param(
            {'offset': {'inclusive': True}},
            r"convention entry for the column 'offset', which the events frame does not have",
            id='offset',
        ),
        pytest.param(
            {'offset_vendor': {'inclusive': False}, 'offset': {'inclusive': True}},
            r"convention entries for the columns 'offset_vendor', 'offset', "
            r'which the events frame does not have',
            id='two_columns',
        ),
    ],
)
def test_metadata_convention_entry_without_column_is_kept_and_warns_once(
        metadata, expected_match,
):
    with pytest.warns(UserWarning, match=expected_match) as record:
        events = Events(onsets=[0], durations=[5], metadata=metadata)

    assert len(record) == 1
    assert events.metadata == metadata


def test_metadata_convention_entry_of_present_extra_column_does_not_warn():
    metadata = {'offset_vendor': {'inclusive': False}}
    data = pl.DataFrame({'onset': [0], 'duration': [5], 'offset_vendor': [5]})

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        events = Events(data, metadata=metadata)

    assert events.metadata == metadata


def test_metadata_convention_entry_without_column_does_not_warn_with_validate_false():
    metadata = {'offset': {'inclusive': True}}

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        events = Events(onsets=[0], durations=[5], metadata=metadata, validate=False)

    assert events.metadata == metadata


@pytest.mark.parametrize('column', ['offset', 'offset_vendor'])
def test_drop_offset_column_removes_its_convention_entry(column):
    data = pl.DataFrame({'onset': [0], 'offset': [5], 'offset_vendor': [4]})
    events = Events(
        data, offsets_inclusive=False, sampling_rate=1000,
        metadata={'offset_vendor': {'inclusive': True}, 'key': 'value'},
    )
    expected_metadata = dict(events.metadata)
    del expected_metadata[column]

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        events.drop(column)

    assert column not in events.columns
    assert events.metadata == expected_metadata


def test_drop_column_removes_the_entry_keyed_by_its_name():
    data = pl.DataFrame({'onset': [0], 'pupil': [3.0]})
    events = Events(data, metadata={'pupil': {'Units': 'mm'}, 'key': 'value'})

    events.drop('pupil')

    assert 'pupil' not in events.columns
    assert events.metadata == {'key': 'value'}

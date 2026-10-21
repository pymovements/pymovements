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
"""Test the input and output contract validation of the generation framework."""
import polars as pl
import pytest

from pymovements import Events
from pymovements import Gaze
from pymovements.synthetic import ConditioningMode
from pymovements.synthetic import GazeGenerator
from pymovements.synthetic import GeneratorCapabilities
from pymovements.synthetic import GeneratorConfig
from pymovements.synthetic import InputContractError
from pymovements.synthetic import OutputContractError
from pymovements.synthetic import SignalRepresentation
from pymovements.synthetic.validation import validate_input
from pymovements.synthetic.validation import validate_output

pytestmark = pytest.mark.filterwarnings(
    'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
)


def make_events(names=('fixation',), onsets=(0,), offsets=(100,)):
    """Build a minimal event sequence."""
    return Events(
        name=list(names),
        onsets=list(onsets),
        offsets=list(offsets),
        time_unit='ms',
    )


def make_gaze(n_samples=4, column='position', with_nulls=False, with_events=False):
    """Build a minimal gaze sequence carrying the requested column group."""
    x = [float(index) for index in range(n_samples)]
    y = list(x)
    if with_nulls and n_samples:
        y[-1] = None

    return Gaze(
        samples=pl.DataFrame({'x': x, 'y': y}, schema={'x': pl.Float64, 'y': pl.Float64}),
        events=make_events() if with_events else None,
        **{f'{column}_columns': ['x', 'y']},
    )


EVENT_CAPABILITIES = GeneratorCapabilities(
    conditioning_modes=frozenset({ConditioningMode.EVENTS}),
    signal_representations=frozenset({SignalRepresentation.POSITION}),
    event_types=frozenset({'fixation'}),
)

UNCONDITIONAL_CAPABILITIES = GeneratorCapabilities(
    conditioning_modes=frozenset({ConditioningMode.UNCONDITIONAL}),
    signal_representations=frozenset({SignalRepresentation.POSITION}),
)

GAZE_CAPABILITIES = GeneratorCapabilities(
    conditioning_modes=frozenset({ConditioningMode.GAZE}),
    signal_representations=frozenset({SignalRepresentation.VELOCITY_2D}),
)

FIXED_RATE_CAPABILITIES = GeneratorCapabilities(
    conditioning_modes=frozenset({ConditioningMode.UNCONDITIONAL}),
    signal_representations=frozenset({SignalRepresentation.POSITION}),
    sampling_rates=frozenset({1000.0}),
    sequence_duration_s=5.0,
    requires_artifacts=True,
)


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(
            {'capabilities': UNCONDITIONAL_CAPABILITIES, 'config': None},
            id='unconditional_mode',
        ),

        pytest.param(
            {'capabilities': EVENT_CAPABILITIES, 'config': None, 'events': make_events()},
            id='event_conditioned_mode',
        ),

        pytest.param(
            {'capabilities': GAZE_CAPABILITIES, 'config': None, 'gaze': make_gaze()},
            id='gaze_conditioned_mode',
        ),

        pytest.param(
            {'capabilities': None, 'config': GeneratorConfig(sampling_rate=1000.0)},
            id='undeclared_capabilities_skip_capability_checks',
        ),

        pytest.param(
            {'capabilities': None, 'config': None, 'events': make_events()},
            id='events_are_not_checked_against_an_absent_declaration',
        ),

        pytest.param(
            {
                'capabilities': GeneratorCapabilities(
                    conditioning_modes=frozenset({ConditioningMode.EVENTS}),
                    signal_representations=frozenset({SignalRepresentation.POSITION}),
                ),
                'config': None,
                'events': make_events(names=('saccade',)),
            },
            id='any_event_type_when_no_event_types_declared',
        ),

        pytest.param(
            {
                'capabilities': FIXED_RATE_CAPABILITIES,
                'config': GeneratorConfig(sampling_rate=1000.0, duration_s=5.0),
                'artifacts': {'weights': 'sha256:abc'},
            },
            id='declared_rate_duration_and_artifacts_all_match',
        ),

        pytest.param(
            {
                'capabilities': EVENT_CAPABILITIES,
                'config': GeneratorConfig(sampling_rate=250.0),
                'events': make_events(
                    names=('fixation', 'fixation'), onsets=(0, 200),
                    offsets=(100, 300),
                ),
            },
            id='arbitrary_sampling_rate_when_none_declared',
        ),
    ],
)
def test_validate_input_returns_none_on_valid(kwargs):
    assert validate_input(**kwargs) is None


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'capabilities': EVENT_CAPABILITIES,
                'config': None,
                'events': make_events(),
                'gaze': make_gaze(),
            },
            InputContractError,
            ('both', 'events', 'gaze', 'one'),
            id='events_and_gaze_together_raise',
        ),

        pytest.param(
            {'capabilities': EVENT_CAPABILITIES, 'config': None},
            InputContractError,
            ('unconditional', 'not supported', 'events'),
            id='no_input_for_event_conditioned_generator_raises',
        ),

        pytest.param(
            {'capabilities': UNCONDITIONAL_CAPABILITIES, 'config': None, 'events': make_events()},
            InputContractError,
            ('events', 'not supported', 'unconditional'),
            id='events_for_unconditional_generator_raises',
        ),

        pytest.param(
            {'capabilities': EVENT_CAPABILITIES, 'config': None, 'gaze': make_gaze()},
            InputContractError,
            ('gaze', 'not supported', 'events'),
            id='gaze_for_event_conditioned_generator_raises',
        ),

        pytest.param(
            {'capabilities': EVENT_CAPABILITIES, 'config': None, 'events': 'fixations'},
            TypeError,
            ('events', 'Events', 'str'),
            id='events_of_wrong_type_raises',
        ),

        pytest.param(
            {'capabilities': GAZE_CAPABILITIES, 'config': None, 'gaze': 'samples'},
            TypeError,
            ('gaze', 'Gaze', 'str'),
            id='gaze_of_wrong_type_raises',
        ),

        pytest.param(
            {
                'capabilities': EVENT_CAPABILITIES,
                'config': None,
                'events': Events(),
            },
            InputContractError,
            ('events', 'empty'),
            id='empty_events_raise',
        ),

        pytest.param(
            {
                'capabilities': EVENT_CAPABILITIES,
                'config': None,
                'events': make_events(names=('saccade',)),
            },
            InputContractError,
            ('saccade', 'not covered', 'fixation'),
            id='undeclared_event_type_raises',
        ),

        pytest.param(
            {
                'capabilities': GAZE_CAPABILITIES,
                'config': None,
                'gaze': make_gaze(n_samples=0),
            },
            InputContractError,
            ('gaze', 'empty'),
            id='empty_gaze_raises',
        ),

        pytest.param(
            {
                'capabilities': GAZE_CAPABILITIES,
                'config': None,
                'gaze': make_gaze(column='velocity'),
            },
            InputContractError,
            ('gaze', 'position', 'pixel'),
            id='gaze_without_position_column_group_raises',
        ),

        pytest.param(
            {
                'capabilities': FIXED_RATE_CAPABILITIES,
                'config': GeneratorConfig(sampling_rate=500.0),
                'artifacts': {'weights': 'sha256:abc'},
            },
            InputContractError,
            ('sampling rate', '500.0', '1000.0'),
            id='unsupported_sampling_rate_raises',
        ),

        pytest.param(
            {
                'capabilities': FIXED_RATE_CAPABILITIES,
                'config': GeneratorConfig(sampling_rate=1000.0, duration_s=2.0),
                'artifacts': {'weights': 'sha256:abc'},
            },
            InputContractError,
            ('duration', '2.0', '5.0'),
            id='unsupported_duration_raises',
        ),

        pytest.param(
            {
                'capabilities': EVENT_CAPABILITIES,
                'config': GeneratorConfig(sampling_rate=1000.0, duration_s=2.0),
                'events': make_events(),
            },
            InputContractError,
            ('duration_s', '2.0', 'events'),
            id='duration_alongside_events_raises',
        ),

        pytest.param(
            {
                'capabilities': None,
                'config': GeneratorConfig(sampling_rate=1000.0, duration_s=2.0),
                'events': make_events(),
            },
            InputContractError,
            ('duration_s', 'events'),
            id='duration_alongside_events_raises_without_declaration',
        ),

        pytest.param(
            {
                'capabilities': FIXED_RATE_CAPABILITIES,
                'config': GeneratorConfig(sampling_rate=1000.0, duration_s=5.0),
            },
            InputContractError,
            ('artifacts', 'requires'),
            id='missing_artifacts_raise',
        ),

        pytest.param(
            {
                'capabilities': FIXED_RATE_CAPABILITIES,
                'config': GeneratorConfig(sampling_rate=1000.0, duration_s=5.0),
                'artifacts': {},
            },
            InputContractError,
            ('artifacts', 'requires'),
            id='empty_artifacts_raise',
        ),
    ],
)
def test_validate_input_raises_error(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        validate_input(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(
            {
                'gaze': make_gaze(),
                'capabilities': UNCONDITIONAL_CAPABILITIES,
                'config': None,
            },
            id='declared_position_is_populated',
        ),

        pytest.param(
            {
                'gaze': make_gaze(column='velocity'),
                'capabilities': GAZE_CAPABILITIES,
                'config': None,
            },
            id='declared_velocity_is_populated',
        ),

        pytest.param(
            {
                'gaze': make_gaze(n_samples=500),
                'capabilities': UNCONDITIONAL_CAPABILITIES,
                'config': GeneratorConfig(sampling_rate=1000.0, duration_s=0.5),
            },
            id='sample_count_matches_rate_times_duration',
        ),

        pytest.param(
            {'gaze': make_gaze(), 'capabilities': None, 'config': None},
            id='undeclared_capabilities_skip_capability_checks',
        ),

        pytest.param(
            {
                'gaze': make_gaze(with_events=True),
                'capabilities': EVENT_CAPABILITIES,
                'config': None,
            },
            id='events_present_when_event_types_declared',
        ),
    ],
)
def test_validate_output_returns_none_on_valid(kwargs):
    assert validate_output(**kwargs) is None


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {
                'gaze': pl.DataFrame({'x': [0.0]}),
                'capabilities': UNCONDITIONAL_CAPABILITIES,
                'config': None,
            },
            OutputContractError,
            ('gaze', 'dataframe'),
            id='generator_returning_something_else_raises',
        ),

        pytest.param(
            {
                'gaze': make_gaze(n_samples=0),
                'capabilities': UNCONDITIONAL_CAPABILITIES,
                'config': None,
            },
            OutputContractError,
            ('no samples',),
            id='empty_output_raises',
        ),

        pytest.param(
            {
                'gaze': make_gaze(column='velocity'),
                'capabilities': UNCONDITIONAL_CAPABILITIES,
                'config': None,
            },
            OutputContractError,
            ('position', 'declared', 'velocity'),
            id='declared_position_but_velocity_emitted_raises',
        ),

        pytest.param(
            {
                'gaze': make_gaze(n_samples=500),
                'capabilities': UNCONDITIONAL_CAPABILITIES,
                'config': GeneratorConfig(sampling_rate=1000.0, duration_s=1.0),
            },
            OutputContractError,
            ('500', '1000', 'sample'),
            id='sample_count_inconsistent_with_rate_and_duration_raises',
        ),

        pytest.param(
            {
                'gaze': make_gaze(with_nulls=True),
                'capabilities': UNCONDITIONAL_CAPABILITIES,
                'config': None,
            },
            OutputContractError,
            ('null', 'position'),
            id='nulls_in_output_raise',
        ),

        pytest.param(
            {
                'gaze': make_gaze(),
                'capabilities': EVENT_CAPABILITIES,
                'config': None,
            },
            OutputContractError,
            ('events', 'fixation'),
            id='missing_events_when_event_types_declared_raises',
        ),
    ],
)
def test_validate_output_raises_error(kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        validate_output(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring.lower() in msg.lower()


class VelocityMagnitudeGenerator(GazeGenerator):
    """A generator emitting velocity magnitudes, which carry no direction."""

    name = 'test_velocity_magnitude'

    capabilities = GeneratorCapabilities(
        conditioning_modes=frozenset({ConditioningMode.UNCONDITIONAL}),
        signal_representations=frozenset({SignalRepresentation.VELOCITY_MAGNITUDE}),
    )

    def _generate(self, config, rng, **kwargs):
        return Gaze(samples=pl.DataFrame({'velocity_magnitude': [12.0, 9.5, 31.0]}))


def test_no_implicit_conversion():
    # A directionless velocity is not one of the Gaze component groups, so it lands in a plain
    # column and Gaze warns that it could not infer any. Asserted rather than silenced, because
    # a generator emitting one is expected to decide for itself whether that warning is useful.
    with pytest.warns(UserWarning, match='no components could be inferred'):
        gaze = VelocityMagnitudeGenerator().generate()

    # The framework adds no positions the generator never produced, and says what it got.
    assert 'position' not in gaze.samples.columns
    assert 'pixel' not in gaze.samples.columns
    assert gaze.metadata['synthetic']['signal_representations'] == ['velocity_magnitude']


class MisbehavingGenerator(GazeGenerator):
    """Declares positions at 1000 Hz but emits velocities, and too few of them."""

    name = 'test_misbehaving'

    capabilities = GeneratorCapabilities(
        conditioning_modes=frozenset({ConditioningMode.UNCONDITIONAL}),
        signal_representations=frozenset({SignalRepresentation.POSITION}),
    )

    def _generate(self, config, rng, **kwargs):
        return make_gaze(n_samples=500, column='velocity')


class EventConditionedGenerator(GazeGenerator):
    """A generator that conditions on an event sequence."""

    name = 'test_event_conditioned'

    capabilities = EVENT_CAPABILITIES

    def _generate(self, config, rng, events=None, **kwargs):
        return make_gaze(n_samples=4 * len(events), with_events=True)


def test_generate_validates_output_of_a_misbehaving_generator():
    with pytest.raises(OutputContractError) as excinfo:
        MisbehavingGenerator().generate(
            config=GeneratorConfig(
                sampling_rate=1000.0,
                duration_s=1.0,
            ),
        )

    msg, = excinfo.value.args
    assert 'test_misbehaving' in msg


def test_generate_validates_input_before_generating():
    with pytest.raises(InputContractError) as excinfo:
        EventConditionedGenerator().generate()

    msg, = excinfo.value.args
    for msg_substring in ('test_event_conditioned', 'unconditional', 'not supported'):
        assert msg_substring.lower() in msg.lower()


def test_generate_passes_conditioning_input_to_private_generate():
    gaze = EventConditionedGenerator().generate(events=make_events())

    assert gaze.metadata['synthetic']['generator'] == 'test_event_conditioned'


def test_generate_records_supplied_artifacts():
    class ArtifactGenerator(EventConditionedGenerator):
        name = 'test_artifacts'

    gaze = ArtifactGenerator().generate(
        events=make_events(),
        artifacts={'weights': 'sha256:abc'},
    )

    assert gaze.metadata['synthetic']['artifacts'] == {'weights': 'sha256:abc'}

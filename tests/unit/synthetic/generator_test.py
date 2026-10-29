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
"""Test the event and sample generator base classes."""
import dataclasses

import numpy as np
import pytest
from polars.testing import assert_frame_equal

from pymovements import Events
from pymovements import Experiment
from pymovements import ExperimentalWarning
from pymovements import Gaze
from pymovements.synthetic import EventGenerator
from pymovements.synthetic import SampleGenerator

pytestmark = pytest.mark.filterwarnings(
    'ignore:Synthetic gaze generation is experimental:pymovements.ExperimentalWarning',
)


@pytest.mark.parametrize(
    'base_class',
    [
        pytest.param(EventGenerator, id='event_generator'),
        pytest.param(SampleGenerator, id='sample_generator'),
    ],
)
def test_base_class_is_abstract(base_class):
    with pytest.raises(TypeError) as excinfo:
        base_class()

    msg, = excinfo.value.args
    for msg_substring in ('abstract', base_class.__name__, '_generate'):
        assert msg_substring in msg


def test_sample_generator_returns_gaze(noise_sample_generator):
    gaze = noise_sample_generator().generate(sampling_rate=1000.0, seed=42)

    assert isinstance(gaze, Gaze)
    assert gaze.samples.height == 10
    assert 'position' in gaze.samples.columns


def test_event_generator_returns_events(uniform_event_generator):
    events = uniform_event_generator().generate(duration=1.0, seed=42)

    assert isinstance(events, Events)
    assert events.frame['offset'].dt.total_milliseconds().max() == 1000


#: Each level's toy generator fixture and the run parameters it needs.
LEVELS = [
    pytest.param('noise_sample_generator', {'sampling_rate': 1000.0}, id='samples'),
    pytest.param('uniform_event_generator', {'duration': 2.0}, id='events'),
]


@pytest.mark.parametrize(
    ('generator', 'run', 'frame'),
    [
        pytest.param('noise_sample_generator', {'sampling_rate': 1000.0}, 'samples', id='samples'),
        pytest.param('uniform_event_generator', {'duration': 2.0}, 'frame', id='events'),
    ],
)
@pytest.mark.parametrize(
    ('first_seed', 'second_seed', 'expected_equal'),
    [
        pytest.param(42, 42, True, id='same_seed_generates_identically'),
        pytest.param(42, 7, False, id='different_seed_generates_differently'),
    ],
)
def test_generator_is_seeded(
        request, generator, run, frame, first_seed, second_seed, expected_equal,
):
    generator = request.getfixturevalue(generator)()

    first = getattr(generator.generate(seed=first_seed, **run), frame)
    second = getattr(generator.generate(seed=second_seed, **run), frame)

    assert first.equals(second) == expected_equal


def test_drawn_seed_reproduces_the_run(noise_sample_generator):
    generator = noise_sample_generator()

    unseeded = generator.generate(sampling_rate=1000.0)
    replayed = generator.generate(
        sampling_rate=1000.0, seed=unseeded.metadata['synthetic']['seed'],
    )

    assert_frame_equal(unseeded.samples, replayed.samples)


def test_sample_generator_attaches_a_copy_of_the_events(noise_sample_generator, fixations):
    gaze = noise_sample_generator().generate(events=fixations, sampling_rate=1000.0, seed=42)

    assert_frame_equal(gaze.events.frame, fixations.frame)
    assert gaze.events is not fixations


def test_sample_generator_conditions_on_the_attached_copy(noise_sample_generator, fixations):
    """A generator that changes its input changes neither the caller's events nor the match."""
    class MutatingGenerator(noise_sample_generator):
        def _generate(self, *, events, **kwargs):
            events.frame = events.frame.head(1)
            return super()._generate(events=events, **kwargs)

    gaze = MutatingGenerator().generate(events=fixations, sampling_rate=1000.0, seed=42)

    assert fixations.frame.height == 2
    assert gaze.events.frame.height == 1
    assert gaze.samples.height == 10


@pytest.mark.parametrize(
    ('sampling_rate', 'seed'),
    [
        pytest.param(np.int64(500), 42, id='numpy_integer_sampling_rate'),
        pytest.param(np.float64(500.0), 42, id='numpy_float_sampling_rate'),
        pytest.param(500.0, np.int64(42), id='numpy_integer_seed'),
        pytest.param(500.0, np.uint32(42), id='numpy_unsigned_seed'),
    ],
)
def test_sample_generator_accepts_numpy_numbers(noise_sample_generator, sampling_rate, seed):
    """The checks accept numpy numbers and record them as plain Python numbers."""
    gaze = noise_sample_generator().generate(sampling_rate=sampling_rate, seed=seed)

    record = gaze.metadata['synthetic']
    assert gaze.experiment.sampling_rate == 500.0
    assert not isinstance(record['run']['sampling_rate'], np.generic)
    assert not isinstance(record['seed'], np.generic)
    assert record['seed'] == 42


def test_sample_generator_attaches_an_experiment_with_the_sampling_rate(noise_sample_generator):
    gaze = noise_sample_generator().generate(sampling_rate=500.0, seed=42)

    assert gaze.experiment.sampling_rate == 500.0


def test_sample_generator_attaches_the_supplied_experiment(noise_sample_generator):
    experiment = Experiment(1280, 1024, 38, 30, 68, 'upper left', 1000.0)

    gaze = noise_sample_generator().generate(
        sampling_rate=1000.0, seed=42, experiment=experiment,
    )

    assert gaze.experiment is experiment


def test_sample_generator_fills_the_sampling_rate_into_an_experiment_without_one(
        noise_sample_generator,
):
    """An experiment built for the screen geometry only gets the requested rate, as a copy."""
    experiment = Experiment(1280, 1024, 38, 30, 68, 'upper left', None)

    gaze = noise_sample_generator().generate(
        sampling_rate=1000.0, seed=42, experiment=experiment,
    )

    assert gaze.experiment.sampling_rate == 1000.0
    assert gaze.experiment.screen == experiment.screen
    assert experiment.sampling_rate is None


def test_sample_generator_keeps_metadata_of_its_own(noise_sample_generator):
    class MetadataGenerator(noise_sample_generator):
        def _generate(self, **kwargs):
            gaze = super()._generate(**kwargs)
            gaze.metadata = {'note': 'kept'}
            return gaze

    gaze = MetadataGenerator().generate(sampling_rate=1000.0, seed=42)

    assert gaze.metadata['note'] == 'kept'
    assert 'synthetic' in gaze.metadata


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'msg_substrings'),
    [
        pytest.param(
            {'sampling_rate': 0.0},
            ValueError,
            ('sampling_rate', 'greater than zero'),
            id='zero_sampling_rate_raises_value_error',
        ),
        pytest.param(
            {'sampling_rate': '1000'},
            TypeError,
            ('sampling_rate', 'number', 'str'),
            id='string_sampling_rate_raises_type_error',
        ),
        pytest.param(
            {'sampling_rate': True},
            TypeError,
            ('sampling_rate', 'number', 'bool'),
            id='bool_sampling_rate_raises_type_error',
        ),
        pytest.param(
            {'sampling_rate': 1000.0, 'experiment': Experiment(sampling_rate=500.0)},
            ValueError,
            ('500.0 Hz', '1000.0 Hz', 'experiment'),
            id='experiment_with_other_sampling_rate_raises_value_error',
        ),
    ],
)
def test_sample_generator_raises_error(noise_sample_generator, kwargs, exception, msg_substrings):
    with pytest.raises(exception) as excinfo:
        noise_sample_generator().generate(**kwargs)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


@pytest.mark.parametrize(('generator', 'run'), LEVELS)
@pytest.mark.parametrize(
    ('seed', 'exception', 'msg_substrings'),
    [
        pytest.param(-1, ValueError, ('seed', 'negative'), id='negative_seed_raises_value_error'),
        pytest.param(
            4.2, TypeError, ('seed', 'integer', 'float'), id='float_seed_raises_type_error',
        ),
    ],
)
def test_generator_raises_error_on_seed(request, generator, run, seed, exception, msg_substrings):
    generator = request.getfixturevalue(generator)()

    with pytest.raises(exception) as excinfo:
        generator.generate(seed=seed, **run)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


@pytest.mark.parametrize(
    ('generator', 'kwargs', 'msg_substring'),
    [
        pytest.param(
            'noise_sample_generator', {}, 'sampling_rate', id='missing_sampling_rate',
        ),
        pytest.param(
            'noise_sample_generator',
            {'sampling_rate': 1000.0, 'sampling_rte': 500.0},
            'sampling_rte',
            id='misspelled_sample_run_parameter',
        ),
        pytest.param(
            'uniform_event_generator', {}, 'duration', id='missing_declared_run_parameter',
        ),
        pytest.param(
            'uniform_event_generator',
            {'duration': 1.0, 'duraton': 2.0},
            'duraton',
            id='misspelled_event_run_parameter',
        ),
    ],
)
def test_generator_rejects_unknown_or_missing_run_parameters(
        request, generator, kwargs, msg_substring,
):
    generator_class = request.getfixturevalue(generator)

    with pytest.raises(TypeError) as excinfo:
        generator_class().generate(**kwargs)

    msg, = excinfo.value.args
    assert msg_substring in msg


@pytest.mark.parametrize(
    ('generator', 'kwargs'),
    [
        pytest.param('noise_sample_generator', {'sampling_rate': 1000.0}, id='sample_generator'),
        pytest.param('uniform_event_generator', {'duration': 1.0}, id='event_generator'),
    ],
)
def test_generate_warns_that_it_is_experimental(request, generator, kwargs):
    generator_class = request.getfixturevalue(generator)

    with pytest.warns(ExperimentalWarning, match='Synthetic gaze generation is experimental'):
        generator_class().generate(**kwargs)


def test_hyperparameters_are_the_dataclass_fields(uniform_event_generator):
    generator = uniform_event_generator(min_duration=50.0)

    assert generator.hyperparameters() == {'min_duration': 50.0, 'max_duration': 300.0}


def test_hyperparameters_leave_out_fields_without_repr(noise_sample_generator):
    @dataclasses.dataclass
    class WeightedGenerator(noise_sample_generator):
        weights: list = dataclasses.field(default_factory=lambda: [0.1, 0.2], repr=False)

    assert WeightedGenerator().hyperparameters() == {'scale': 1.0}


class _NeverGenerates(SampleGenerator):
    """Fails if it gets to synthesize, so a rejection is shown to happen before ``_generate``."""

    def _generate(self, **run):
        raise AssertionError('_generate must not run for a generator that is rejected')


class _PlainGenerator(_NeverGenerates):
    name = 'plain'


@dataclasses.dataclass
class _UnnamedGenerator(_NeverGenerates):
    pass


@dataclasses.dataclass
class _EmptyNameGenerator(_NeverGenerates):
    name = ''


@dataclasses.dataclass
class _AnnotatedNameGenerator(_NeverGenerates):
    name: str = 'annotated'  # type: ignore[misc]


@pytest.mark.parametrize(
    ('generator_class', 'msg_substrings'),
    [
        pytest.param(
            _PlainGenerator,
            ('_PlainGenerator', 'dataclass', 'reproduced'),
            id='not_a_dataclass',
        ),
        pytest.param(
            _UnnamedGenerator,
            ('_UnnamedGenerator', "'name'", 'class attribute'),
            id='name_missing',
        ),
        pytest.param(
            _EmptyNameGenerator,
            ('_EmptyNameGenerator', "'name'", 'non-empty'),
            id='name_empty',
        ),
        pytest.param(
            _AnnotatedNameGenerator,
            ('_AnnotatedNameGenerator', "'name'", 'annotation', 'hyperparameter'),
            id='name_declared_as_a_field',
        ),
    ],
)
def test_generate_rejects_a_generator_class_that_cannot_be_recorded(
        generator_class, msg_substrings,
):
    with pytest.raises(TypeError) as excinfo:
        generator_class().generate(sampling_rate=1000.0, seed=42)

    msg, = excinfo.value.args
    for msg_substring in msg_substrings:
        assert msg_substring in msg


def test_hyperparameters_reject_a_generator_that_is_no_dataclass():
    with pytest.raises(TypeError) as excinfo:
        _PlainGenerator().hyperparameters()

    msg, = excinfo.value.args
    for msg_substring in ('_PlainGenerator', 'dataclass'):
        assert msg_substring in msg

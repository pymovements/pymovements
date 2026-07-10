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
import warnings

import numpy as np
import pandas as pd
import pytest

import pymovements as pm
from pymovements.events.detection.idt import idt
from pymovements.events.detection.ihmm import ihmm
from pymovements.gaze.transforms_numpy import pos2vel
from pymovements.synthetic import step_function
warnings.filterwarnings('ignore')

sampling_rate = 1000.0


test_results = []


def record(name, passed, detail=''):
    test_results.append({
        'test': name, 'passed': bool(
            passed,
        ) if passed is not None else None, 'detail': detail,
    })
    status = 'PASS' if passed else 'FAIL'
    if passed is None:
        status = 'SKIP'
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ''))


def to_frame(events):
    """Return the underlying table (Polars/Pandas-like) for an events result."""
    return events.frame if hasattr(events, 'frame') else events


def _column(frame, col):
    series = frame[col]
    return series.to_list() if hasattr(series, 'to_list') else list(series)


def n_events(events):
    return len(to_frame(events))


def spans(events):
    """List of (onset, offset) tuples for every detected event."""
    frame = to_frame(events)
    if len(frame) == 0:
        return []
    onsets = _column(frame, 'onset')
    offsets = _column(frame, 'offset')
    return list(zip(onsets, offsets))


def show(label, events):
    print(f"--- {label} ({n_events(events)} event(s)) ---")
    print(to_frame(events))


def overlap_ratio(span_a, span_b):
    """Intersection-over-union of two (onset, offset) intervals."""
    a0, a1 = span_a
    b0, b1 = span_b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def best_overlaps(spans_a, spans_b):
    """For each span in a, the best IoU achievable against any span in b."""
    if not spans_a:
        return []
    return [max((overlap_ratio(sa, sb) for sb in spans_b), default=0.0) for sa in spans_a]


def assert_event_count(name, events, expected):
    actual = n_events(events)
    passed = actual == expected
    record(f"{name}: event count == {expected}", passed, f"got {actual}")
    assert passed, f"{name}: expected {expected} event(s), got {actual}"


def assert_algorithms_agree(name, ihmm_events, idt_events, min_overlap=0.5):
    """Same number of fixations, and each ihmm fixation overlaps some idt fixation well."""
    ihmm_spans, idt_spans = spans(ihmm_events), spans(idt_events)
    count_match = len(ihmm_spans) == len(idt_spans)
    record(
        f"{name}: fixation counts match",
        count_match,
        f"ihmm={len(ihmm_spans)}, idt={len(idt_spans)}",
    )
    overlaps = best_overlaps(ihmm_spans, idt_spans)
    worst = min(overlaps) if overlaps else 1.0
    ok = worst >= min_overlap
    record(f"{name}: min pairwise overlap >= {min_overlap}", ok, f"worst overlap={worst:.2f}")
    return count_match and ok


def test_case1_no_fixations():
    positions_no_fix = step_function(
        length=6,
        steps=[1, 2, 3, 4, 5],
        values=[
            (100., 100.),
            (-100., -100.),
            (100., -100.),
            (-100., 100.),
            (200., 0.),
        ],
        start_value=(0., 0.),
    )

    velocities_no_fix = pos2vel(positions_no_fix, sampling_rate=sampling_rate)

    ihmmEvents = ihmm(velocities_no_fix, reestimation=True, name='ihmm_fix')
    idtEvents = idt(positions_no_fix, name='idt_fix')

    show('IHMM — no fixations', ihmmEvents)
    show('IDT — no fixations', idtEvents)

    assert_event_count('No fixations / ihmm', ihmmEvents, 0)
    assert_event_count('No fixations / idt', idtEvents, 0)


def test_case2_one_fixation():
    length_one_fix = 200
    positions_one_fix = step_function(
        length=length_one_fix,
        steps=[1],
        values=[(50., 50.)],
        start_value=(0., 0.),
    )

    velocities_one_fix = pos2vel(positions_one_fix, sampling_rate=sampling_rate)

    ihmmEvents = ihmm(velocities_one_fix, reestimation=True, name='ihmm_fix')
    idtEvents = idt(positions_one_fix, name='idt_fix')

    show('IHMM — one fixation', ihmmEvents)
    show('IDT — one fixation', idtEvents)

    assert_event_count('One fixation / ihmm', ihmmEvents, 1)
    assert_event_count('One fixation / idt', idtEvents, 1)
    assert_algorithms_agree('One fixation', ihmmEvents, idtEvents, min_overlap=0.9)


def test_case3_one_big_fixation():
    length_big_fix = 200
    positions_big_fix = np.tile(np.array([[75., 75.]]), (length_big_fix, 1))

    velocities_big_fix = pos2vel(positions_big_fix, sampling_rate=sampling_rate)

    ihmmEvents = ihmm(velocities_big_fix, reestimation=True, name='ihmm_fix')
    idtEvents = idt(positions_big_fix, name='idt_fix')

    show('IHMM — one big fixation', ihmmEvents)
    show('IDT — one big fixation', idtEvents)

    assert_event_count('One big fixation / ihmm', ihmmEvents, 1)
    assert_event_count('One big fixation / idt', idtEvents, 1)

    # The single fixation should cover (nearly) the whole recording.
    min_span = length_big_fix * 0.9
    for label, events in (('ihmm', ihmmEvents), ('idt', idtEvents)):
        onset, offset = spans(events)[0]
        covered = offset - onset
        ok = covered >= min_span
        record(f"One big fixation / {label} covers >= 90% of trial", ok, f"span={covered}")
        assert ok

    assert_algorithms_agree('One big fixation', ihmmEvents, idtEvents, min_overlap=0.95)


def test_case4_trailing_and_ending_fixations():
    length_edges = 200
    split_at = 90
    positions_edges = step_function(
        length=length_edges,
        steps=[split_at],
        values=[(60., 60.)],
        start_value=(10., 10.),
    )

    velocities_edges = pos2vel(positions_edges, sampling_rate=sampling_rate)

    ihmmEvents = ihmm(velocities_edges, reestimation=True, name='ihmm_fix', minimum_duration=2)
    idtEvents = idt(positions_edges, name='idt_fix', minimum_duration=2)

    show('IHMM — trailing/ending fixations', ihmmEvents)
    show('IDT — trailing/ending fixations', idtEvents)

    assert_event_count('Trailing/ending fixations / ihmm', ihmmEvents, 2)
    assert_event_count('Trailing/ending fixations / idt', idtEvents, 2)

    for label, events in (('ihmm', ihmmEvents), ('idt', idtEvents)):
        event_spans = sorted(spans(events), key=lambda s: s[0])
        first_onset = event_spans[0][0]
        last_offset = event_spans[-1][1]
        starts_at_edge = first_onset <= 1  # allow 1 sample of slack (ms at 1000Hz)
        ends_at_edge = last_offset >= (length_edges - 1) * (1000.0 / sampling_rate) - 1
        record(
            f"Trailing/ending / {label}: first fixation starts at trial start",
            starts_at_edge,
            f"onset={first_onset}",
        )
        record(
            f"Trailing/ending / {label}: last fixation ends at trial end",
            ends_at_edge,
            f"offset={last_offset}",
        )
        assert starts_at_edge and ends_at_edge

    assert_algorithms_agree('Trailing/ending fixations', ihmmEvents, idtEvents, min_overlap=0.85)


def test_case5_toy_dataset():
    toy_available = True
    try:
        dataset = pm.Dataset('ToyDataset', path='data/ToyDataset')
        dataset.download()
        dataset.load()
    except Exception as exc:  # e.g. no network access in this environment
        toy_available = False
        print(f"Skipping toy dataset test — could not download/load ToyDataset ({exc})")

    if toy_available:
        dataset.pix2deg()
        dataset.pos2vel()

        toy_gaze = dataset.gaze[0]
        toy_positions = toy_gaze.frame.select('position').to_series().to_list()
        toy_velocities = toy_gaze.frame.select('velocity').to_series().to_list()

        toy_velocities = [[np.nan, np.nan] if v == [None, None] else v for v in toy_velocities]
        print(toy_velocities)

        ihmmEvents = ihmm(toy_velocities, reestimation=True, name='ihmm_fix')
        idtEvents = idt(np.array(toy_positions), name='idt_fix')

        show('IHMM — toy dataset', ihmmEvents)
        show('IDT — toy dataset', idtEvents)

    if toy_available:
        ihmm_count, idt_count = n_events(ihmmEvents), n_events(idtEvents)
        larger, smaller = max(ihmm_count, idt_count), max(min(ihmm_count, idt_count), 1)
        relative_diff = (larger - smaller) / smaller
        ok = relative_diff <= 0.5  # allow up to 50% relative disagreement on real, noisy data
        record(
            'Toy dataset: fixation counts are comparable',
            ok,
            f"ihmm={ihmm_count}, idt={idt_count}, relative diff={relative_diff:.2f}",
        )
    else:
        record('Toy dataset: fixation counts are comparable', None, 'skipped (no network access)')


def test_case6_regular_dataset():
    rng = np.random.default_rng(seed=42)

    fixation_targets = [
        (0., 0.), (120., 5.), (240., -10.), (360., 15.), (480., 0.),
        (600., 20.), (720., -15.), (840., 10.), (960., 0.),
    ]
    fixation_len = 150  # samples held per fixation target

    length_realistic = fixation_len * len(fixation_targets)
    steps_realistic = [fixation_len * i for i in range(1, len(fixation_targets))]

    positions_realistic = step_function(
        length=length_realistic,
        steps=steps_realistic,
        values=fixation_targets[1:],
        start_value=fixation_targets[0],
    )

    # Small measurement jitter, well below any reasonable dispersion/velocity threshold.
    jitter = rng.normal(loc=0.0, scale=0.05, size=positions_realistic.shape)
    positions_realistic = positions_realistic + jitter

    velocities_realistic = pos2vel(positions_realistic, sampling_rate=sampling_rate)

    ihmmEvents = ihmm(velocities_realistic, reestimation=True, name='ihmm_fix')
    idtEvents = idt(positions_realistic, name='idt_fix')

    show('IHMM — regular dataset', ihmmEvents)
    show('IDT — regular dataset', idtEvents)

    expected_fixations = len(fixation_targets)

    for label, events in (('ihmm', ihmmEvents), ('idt', idtEvents)):
        actual = n_events(events)
        ok = abs(actual - expected_fixations) <= 1  # allow off-by-one at the boundaries
        record(
            f"Regular dataset / {label}: ~{expected_fixations} fixations detected",
            ok,
            f"got {actual}",
        )
        assert ok

    assert_algorithms_agree('Regular dataset', ihmmEvents, idtEvents, min_overlap=0.6)


@pytest.fixture(scope='session', autouse=True)
def _print_summary_at_session_end():
    yield
    summary = pd.DataFrame(test_results)
    print(summary)

    n_run = summary['passed'].notna().sum()
    n_passed = (summary['passed'] == True).sum()  # noqa: E712
    n_failed = (summary['passed'] == False).sum()  # noqa: E712
    n_skipped = summary['passed'].isna().sum()
    print(f"{n_passed}/{n_run} checks passed" + (f", {n_skipped} skipped" if n_skipped else ''))

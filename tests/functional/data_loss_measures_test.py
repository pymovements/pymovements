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
"""Test measure-based replacements for the removed data-loss metadata fields.

The ``data_loss_ratio`` and ``data_loss_ratio_blinks`` fields were removed from
the EyeLink parser metadata in favor of ``pymovements.measure.data_loss`` and
``Gaze.measure_events_ratio`` (see issue #1584).
"""
import pytest

from pymovements.gaze import from_asc


@pytest.mark.parametrize(
    'expected_ratio',
    [
        pytest.param(
            # durations are summed without merging the overlap:
            # ((1101 - 1005 + 1) + (1090 - 1011 + 1)) / (1104 - 1000 + 1) = 177 / 105
            177 / 105,
            id='current_behavior_overlap_double_counted',
        ),
        pytest.param(
            # merging the overlapping intervals yields (1101 - 1005 + 1) / 105 = 97 / 105,
            # the value the removed data_loss_ratio_blinks metadata field reported
            97 / 105,
            marks=pytest.mark.xfail(
                reason='overlapping events are not merged before summing durations (#1584)',
                strict=True,
            ),
            id='expected_behavior_overlap_merged',
        ),
    ],
)
@pytest.mark.filterwarnings('ignore:.*No eye tracker vendor found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No eye tracker model found.*:UserWarning')
@pytest.mark.filterwarnings('ignore:.*No eye tracker software version found.*:UserWarning')
def test_from_asc_binocular_overlapping_blinks_events_ratio(make_text_file, expected_ratio):
    """Overlapping binocular blink events are double-counted by measure_events_ratio.

    Binocular recordings emit separate left-eye and right-eye blink events which
    typically overlap in time. ``measure_events_ratio`` sums the durations of all
    ``blink_eyelink`` events without merging overlapping intervals, so the ratio
    can exceed 1.0.

    The removed ``data_loss_ratio_blinks`` metadata field merged overlapping blink
    intervals before counting and reported 97 / 105 for this scenario, so the
    suggested migration to ``measure_events_ratio`` is not equivalent for
    binocular data (see issue #1584). The xfailing parametrization asserts the
    correct merged result and is to be addressed in a follow-up PR.
    """
    start = 1000
    end = 1104

    asc_lines = [
        'MSG\t990 RECCFG CR 1000 2 1 LR',
        'MSG\t990 ELCLCFG BTABLER',
        'MSG\t990 GAZE_COORDS 0.00 0.00 1919.00 1079.00',
        'PUPIL\tAREA',
        'EVENTS\tGAZE\tLEFT\tRIGHT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2',
        'SAMPLES\tGAZE\tLEFT\tRIGHT\tRATE\t1000.00\tTRACKING\tCR\tFILTER\t2',
        f'START\t{start}\tLEFT\tRIGHT\tSAMPLES\tEVENTS',
    ]

    # left-eye blink [1005, 1101] fully contains right-eye blink [1011, 1090]
    for t in range(start, end + 1):
        if t == 1005:
            asc_lines.append(f'SBLINK L {t}')
        if t == 1011:
            asc_lines.append(f'SBLINK R {t}')
        if t == 1091:
            asc_lines.append('EBLINK R 1011\t1090\t80')
        if t == 1102:
            asc_lines.append('EBLINK L 1005\t1101\t97')

        if 1005 <= t <= 1101:
            asc_lines.append(f'{t}\t  .\t  .\t   0.0\t  .\t  .\t   0.0\t.C.C.')
        else:
            asc_lines.append(f'{t}\t 966.9\t 565.5\t 276.0\t 949.4\t 545.5\t 308.0\t.....')

    asc_lines.append(f'END\t{end}\tSAMPLES\tEVENTS\tRES\t 47.75\t 45.92')

    filepath = make_text_file(
        filename='binocular_overlapping_blinks.asc',
        body='\n'.join(asc_lines) + '\n',
    )

    gaze = from_asc(filepath, events=True)

    ratio = gaze.samples.select(
        gaze.measure_events_ratio('blink_eyelink', sampling_rate=1000.0),
    ).item()

    assert ratio == pytest.approx(expected_ratio)

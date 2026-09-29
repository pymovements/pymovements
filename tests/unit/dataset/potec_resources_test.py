# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Test the resource definitions of the PoTeC dataset."""
from pathlib import Path

import pytest

from pymovements import DatasetDefinition
from pymovements import DatasetLibrary
from pymovements import ResourceDefinition
from pymovements._utils._strings import curly_to_regex
from pymovements.stimulus import text as text_stimulus

EXAMPLE_AOI_FILE = Path(__file__).parent.parent.parent / 'files' / 'potec_char_aoi_example.ias'


@pytest.fixture(name='potec')
def fixture_potec() -> DatasetDefinition:
    """Return the PoTeC dataset definition.

    Returns
    -------
    DatasetDefinition
        The definition registered in the dataset library.

    """
    return DatasetLibrary.get('PoTeC')


def resource(
        definition: DatasetDefinition,
        content: str,
        needle: str,
) -> ResourceDefinition:
    """Return the single resource of a content type whose filename pattern contains a needle.

    Parameters
    ----------
    definition: DatasetDefinition
        The dataset definition to search.
    content: str
        Content type of the resource.
    needle: str
        Substring that identifies the resource among those of the same content type.

    Returns
    -------
    ResourceDefinition
        The matching resource.

    """
    matches = [
        candidate for candidate in definition.resources
        if candidate.content == content and needle in candidate.filename_pattern
    ]
    assert len(matches) == 1, f'expected exactly one {content} resource matching {needle!r}'
    return matches[0]


def test_potec_has_corrected_and_uncorrected_fixations(potec):
    patterns = [
        candidate.filename_pattern for candidate in potec.resources
        if candidate.content == 'precomputed_events'
    ]

    assert len(patterns) == 2
    assert any('uncorrected' in pattern for pattern in patterns)
    assert any('uncorrected' not in pattern for pattern in patterns)


@pytest.mark.parametrize(
    ('needle', 'filename', 'expected'),
    [
        pytest.param(
            'uncorrected',
            'reader18_b1_uncorrected_fixations.tsv',
            {'subject_id': '18', 'text_id': 'b1'},
            id='uncorrected_matches_its_own',
        ),
        pytest.param(
            'uncorrected',
            'reader18_b1_fixations.tsv',
            None,
            id='uncorrected_rejects_corrected',
        ),
        pytest.param(
            '_fixations',
            'reader18_b1_fixations.tsv',
            {'subject_id': '18', 'text_id': 'b1'},
            id='corrected_matches_its_own',
        ),
        pytest.param(
            '_fixations',
            'reader18_b1_uncorrected_fixations.tsv',
            None,
            id='corrected_rejects_uncorrected',
        ),
    ],
)
def test_potec_fixation_patterns_do_not_overlap(potec, needle, filename, expected):
    """Both fixation resources land in the same directory, so their patterns must be disjoint.

    With a greedy ``{text_id}`` the corrected pattern also matches the uncorrected filenames,
    capturing ``text_id='b1_uncorrected'``, and every uncorrected trial is scanned twice. The
    two-character length in the pattern is what prevents that.
    """
    if needle == 'uncorrected':
        candidates = [
            candidate for candidate in potec.resources
            if candidate.content == 'precomputed_events' and 'uncorrected' in
            candidate.filename_pattern
        ]
    else:
        candidates = [
            candidate for candidate in potec.resources
            if candidate.content == 'precomputed_events' and 'uncorrected' not in
            candidate.filename_pattern
        ]
    regex = curly_to_regex(candidates[0].filename_pattern)

    match = regex.fullmatch(filename)

    assert (match.groupdict() if match else None) == expected


def test_potec_has_a_character_aoi_resource(potec):
    stimuli = [
        candidate for candidate in potec.resources if candidate.content == 'textstimulus'
    ]

    assert len(stimuli) == 1
    assert stimuli[0].load_kwargs['aoi_column'] == 'character'
    assert stimuli[0].load_kwargs['read_csv_kwargs']['quote_char'] is None


def test_potec_aoi_pattern_matches_the_published_text_ids(potec):
    regex = curly_to_regex(resource(potec, 'textstimulus', '.ias').filename_pattern)
    text_ids = [f'{prefix}{index}' for prefix in 'bp' for index in range(6)]

    matched = [regex.fullmatch(f'{text_id}.ias') for text_id in text_ids]

    assert all(match is not None for match in matched)
    assert [match.group('text_id') for match in matched] == text_ids


def test_potec_aoi_load_kwargs_read_the_published_file(potec):
    load_kwargs = dict(resource(potec, 'textstimulus', '.ias').load_kwargs)
    read_csv_kwargs = load_kwargs.pop('read_csv_kwargs')

    stimulus = text_stimulus.from_file(
        EXAMPLE_AOI_FILE, custom_read_kwargs=read_csv_kwargs, **load_kwargs,
    )

    assert stimulus.aois.height == 185
    assert stimulus.aois['character'].to_list()[:3] == ['U', 'm', 'd']


def test_potec_derived_line_index_matches_the_published_line_column(potec):
    """The .ias files carry a one-based line column; the derived index must agree with it.

    That agreement is all this checks. Within a PoTeC line every box shares one top edge, so
    the file does not exercise the case that centre grouping is chosen for; the height
    differences are between lines, not inside them.
    """
    load_kwargs = dict(resource(potec, 'textstimulus', '.ias').load_kwargs)
    read_csv_kwargs = load_kwargs.pop('read_csv_kwargs')
    stimulus = text_stimulus.from_file(
        EXAMPLE_AOI_FILE, custom_read_kwargs=read_csv_kwargs, **load_kwargs,
    )

    aois = stimulus.with_line_idx().aois

    assert (aois['line_idx'] == aois['line'] - 1).all()


def test_potec_aoi_file_needs_a_disabled_quote_character(tmp_path):
    """A double quote as the character of an AOI must not be read as a quoted field."""
    path = tmp_path / 'q0.ias'
    path.write_text(
        'aoi_type\taoi\tstart_x\tstart_y\tend_x\tend_y\tcharacter\tline\n'
        '0 RECTANGLE\t1\t80\t21\t93\t99\t"\t1\n'
        '0 RECTANGLE\t2\t93\t21\t115\t99\tm\t1\n',
    )

    stimulus = text_stimulus.from_file(
        path, aoi_column='character',
        start_x_column='start_x', start_y_column='start_y',
        end_x_column='end_x', end_y_column='end_y',
        custom_read_kwargs={'separator': '\t', 'quote_char': None},
    )

    assert stimulus.aois['character'].to_list() == ['"', 'm']

    with pytest.raises(ValueError, match='not a valid CSV file'):
        text_stimulus.from_file(
            path, aoi_column='character',
            start_x_column='start_x', start_y_column='start_y',
            end_x_column='end_x', end_y_column='end_y',
            custom_read_kwargs={'separator': '\t'},
        )

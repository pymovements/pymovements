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
"""Validate reading measures against the published PoTeC reading measure files.

PoTeC (Jakobi et al., 2024) publishes word-level reading measures computed by the reference
implementation that pymovements' original ``compute_reading_measures`` descends from. This test
recomputes the measures from the published fixation sequences and compares them cell by cell.

The reference implementation handles the start and end of the fixation sequence differently
from pymovements, so the comparison replicates or masks three known differences:

* The reference never processes the final fixation of a trial, so it is dropped from the input.
* ``SL_in`` of the first fixated word equals the word position instead of 0 (the reference
  starts from a ``-1`` sentinel) and is masked.
* ``FRT``, ``SL_out``, and ``TRC_out`` of the last fixated words depend on how the sequence end
  is handled and are masked.

Everything else must match exactly.

TODO: Clarify with Deborah whether these are deliberate choices of the reference implementation
rather than artifacts. Dropping the final fixation in particular might be reasonable, in which
case pymovements should adopt the behavior instead of masking it here.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pymovements.dataset._utils._archives import extract_archive
from pymovements.dataset.websource import WebSource
from pymovements.measure.reading import compute_reading_measures

SCANPATHS_SOURCE = WebSource(
    url='https://osf.io/download/7fkze/',
    filename='scanpaths_merged.zip',
    md5='a27f74d94cb00946c5f40b3bb54e3705',
)

READING_MEASURES_SOURCE = WebSource(
    url='https://osf.io/download/3ywhz/',
    filename='reading_measures_merged.zip',
    md5='b7ada7ca91f3a807d873598b821de88d',
)

MEASURE_COLUMNS = [
    'FFD', 'SFD', 'FD', 'FPRT', 'FRT', 'TFT', 'RRT', 'RPD_inc', 'RPD_exc',
    'RBRT', 'Fix', 'FPF', 'RR', 'FPReg', 'TRC_out', 'TRC_in', 'SL_in', 'SL_out', 'TFC',
]

# Measures that are masked at the last fixated words (see module docstring).
SEQUENCE_END_MEASURES = ('FRT', 'SL_out', 'TRC_out')

# Every n-th of the 900 published reader/text pairs is compared.
SAMPLE_STRIDE = 60


@pytest.fixture(scope='module', name='potec_directory')
def fixture_potec_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp('potec')
    for source in (SCANPATHS_SOURCE, READING_MEASURES_SOURCE):
        archive_path = source.download(directory, verbose=False)
        extract_archive(
            archive_path, directory / archive_path.stem, remove_finished=True, verbose=0,
        )
    return directory


@pytest.mark.network
def test_reading_measures_match_published_potec_measures(potec_directory):
    scanpath_files = sorted(
        path for path in (potec_directory / 'scanpaths_merged').glob('**/*_merged_sp_rm.tsv')
        if not path.name.startswith('._')  # skip macOS resource fork files in the archive
    )[::SAMPLE_STRIDE]
    assert scanpath_files

    mismatches = []
    for scanpath_file in scanpath_files:
        stem = scanpath_file.name.replace('_merged_sp_rm.tsv', '')
        measure_files = [
            path
            for path in (potec_directory / 'reading_measures_merged').glob(f'**/{stem}_merged.tsv')
            if not path.name.startswith('._')
        ]
        assert len(measure_files) == 1, stem

        scanpaths = pl.read_csv(
            scanpath_file, separator='\t', null_values='.', infer_schema_length=10000,
        )
        expected = pl.read_csv(
            measure_files[0], separator='\t', null_values='.', infer_schema_length=10000,
        ).sort('word_index_in_text')

        fixations = scanpaths.sort('fixation_index').select(
            pl.col('word_index_in_text').alias('word_idx'),
            pl.col('fixation_duration').alias('duration'),
        )
        # The reference implementation never processed the trial's final fixation.
        fixations = fixations.head(fixations.height - 1)

        aois = expected.select(pl.col('word_index_in_text').alias('word_idx'), 'word')

        result = compute_reading_measures(fixations, aois).sort('word_index')
        assert result.height == expected.height, stem

        word_sequence = scanpaths.sort('fixation_index')['word_index_in_text'].drop_nulls()
        first_word = word_sequence[0]
        last_words = set(word_sequence.tail(2))

        for result_row, expected_row in zip(result.to_dicts(), expected.to_dicts()):
            word = result_row['word_index']
            assert result_row['word'] == expected_row['word'], (stem, word)
            for column in MEASURE_COLUMNS:
                if column == 'SL_in' and word == first_word:
                    continue
                if column in SEQUENCE_END_MEASURES and word in last_words:
                    continue
                if result_row[column] != expected_row[column]:
                    mismatches.append(
                        (stem, word, column, result_row[column], expected_row[column]),
                    )

    assert not mismatches, mismatches[:20]

# Copyright (c) 2024-2025 The pymovements Project Authors
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
"""Reading measure tests."""
import polars as pl
import pytest

import pymovements as pm


@pytest.mark.parametrize(
    (
        'detection_method', 'minimum_duration', 'dispersion_threshold',
        'velocity_threshold', 'expected_event_properties', 'use_save_path',
    ),
    [
        pytest.param(
            'ivt', 100, None, 20.0, ('location', {'position_column': 'pixel'}), False,
            id='ivt_detection_no_csv',
        ),
        pytest.param(
            'idt', 100, 1.0, None, ('location', {'position_column': 'pixel'}), True,
            id='idt_detection_save_csv',
        ),
    ],
)
def test_reading_measures_processing(
    detection_method, minimum_duration, dispersion_threshold,
    velocity_threshold, expected_event_properties, use_save_path, tmp_path,
):
    # Create the dataset
    dataset = pm.Dataset('PoTeC', path='data/PoTeC')
    dataset.load(subset={'subject_id': 5, 'text_id': 'b0'})
    dataset.pix2deg()
    dataset.pos2vel()

    # Set detection parameters
    detection_params = {'minimum_duration': minimum_duration}
    if detection_method == 'ivt':
        detection_params['velocity_threshold'] = velocity_threshold
    elif detection_method == 'idt':
        detection_params['dispersion_threshold'] = dispersion_threshold

    # Perform event detection
    dataset.detect(detection_method, **detection_params)
    dataset.compute_event_properties(expected_event_properties)

    # Create the ReadingMeasures object
    reading_measures = pm.reading_measures.ReadingMeasures()
    aoi_dict = {'b0': 'tests/files/potec_word_aoi_b0.tsv'}

    # Determine the save path based on the use_save_path parameter
    save_path = tmp_path if use_save_path else None

    # Process the dataset and potentially save to CSV
    reading_measures.process_dataset(dataset, aoi_dict, save_path=save_path)

    # Example of an expected DataFrame schema check (adjust as needed)
    expected_columns = [
        'word_index', 'word', 'subject_id', 'text_id', 'FFD', 'SFD', 'FD', 'FPRT', 'FRT',
        'TFT', 'RRT', 'RPD_inc', 'RPD_exc', 'RBRT', 'Fix', 'FPF', 'RR', 'FPReg', 'TRC_out',
        'TRC_in', 'SL_in', 'SL_out', 'TFC',
    ]
    result_frame = reading_measures.frame[0]

    # Check that the resulting DataFrame has the expected columns
    assert set(result_frame.columns) == set(
        expected_columns,
    ), 'The result DataFrame does not contain the expected columns.'

    # Example of validating a specific property (e.g., checking that FFD is computed correctly)
    assert result_frame['FFD'].sum() > 0, 'FFD should be greater than zero for valid fixation data.'

    # If use_save_path is True, check that the file is saved correctly
    if use_save_path:
        # Construct the expected file path
        expected_file = save_path / '5-b0-reading_measures.csv'
        assert expected_file.is_file(), f"The CSV file should have been saved at {expected_file}."
        # Optionally, read back the saved file and verify its contents if necessary
        saved_df = pl.read_csv(str(expected_file))
        assert set(saved_df.columns) == set(
            expected_columns,
        ), 'The saved CSV file does not have the expected columns.'

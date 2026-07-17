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
"""Module for the Reading Measure DataFrame."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
from tqdm import tqdm

from pymovements._utils._html import repr_html
from pymovements.measure.reading.processing import compute_reading_measures
from pymovements.stimulus import text

if TYPE_CHECKING:
    import pymovements as pm


@repr_html()
class ReadingMeasures:
    """A DataFrame for reading measures.

    Parameters
    ----------
    reading_measure_df: pl.DataFrame | None
        A reading measure dataframe. (default: None)
    """

    def __init__(self, reading_measure_df: pl.DataFrame | None = None) -> None:
        self.frame: pl.DataFrame
        if reading_measure_df is None:
            self.frame = pl.DataFrame()
        else:
            self.frame = reading_measure_df

    def process_dataset(
        self, dataset: pm.Dataset,
        aoi_dict: dict[str, str | Path], save_path: str | Path | None,
    ) -> int:
        """Map fixations to AOIs and compute reading measures for an entire dataset.

        Parameters
        ----------
        dataset : pm.Dataset
            The dataset containing the events to be processed.
        aoi_dict : dict[str, str | Path]
            A dictionary mapping text IDs to their corresponding AOI file paths.
        save_path : str | Path | None
            The directory path where the computed reading measures CSV files will be saved.
            If ``None``, no files are saved to disk.

        Returns
        -------
        int
            Returns 0 upon successful processing of the dataset.
        """
        for event_idx in tqdm(range(len(dataset.events))):
            tmp_df = dataset.events[event_idx]
            if tmp_df.frame.is_empty():
                print('+ skip due to empty DF')
                continue
            text_id = tmp_df['text_id'][0]
            aoi_text_stimulus = text.from_file(
                aoi_dict[text_id],
                aoi_column='character',
                start_x_column='start_x',
                start_y_column='start_y',
                end_x_column='end_x',
                end_y_column='end_y',
                page_column='page',
                custom_read_kwargs={'separator': '\t'},
            )

            dataset.events[event_idx].map_to_aois(aoi_text_stimulus)

        for _fix_file in dataset.events:
            if _fix_file.frame.is_empty():
                print('+ skip due to empty DF')
                continue
            fixations_df = _fix_file.frame.to_pandas()

            text_id = fixations_df.iloc[0]['text_id']
            subject_id = int(fixations_df.iloc[0]['subject_id'])
            aoi_df = pd.read_csv(aoi_dict[text_id], delimiter='\t')

            rm_df = compute_reading_measures(
                fixations_df=fixations_df,
                aoi_df=aoi_df,
            )

            rm_df['subject_id'] = subject_id
            rm_df['text_id'] = text_id

            # Append the computed reading measures DataFrame to the list
            self.frame.append(rm_df)

            # Save to CSV if save_path is provided
            if save_path is not None:
                rm_filename = f'{subject_id}-{text_id}-reading_measures.csv'
                path_save_rm_file = os.path.join(save_path, rm_filename)
                rm_df.to_csv(path_save_rm_file, index=False)

        return 0

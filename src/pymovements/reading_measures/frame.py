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
"""Module for the Reading Measure DataFrame."""
from __future__ import annotations

import os

import pandas as pd
import polars as pl
from tqdm import tqdm

from pymovements.stimulus import text

from pymovements._utils._html import repr_html


@repr_html()
class ReadingMeasures:
    """A DataFrame for reading measures.

    Parameters
    ----------
    reading_measure_df: pl.DataFrame | None
        A reading measure dataframe. (default: None)
    """

    def __init__(self, reading_measure_df: pl.DataFrame | None = None) -> None:
        self.frame = reading_measure_df
        if reading_measure_df is None:
            self.frame = []

    def process_dataset(self, dataset, aoi_dict, save_path) -> int:
        """Process dataset.

        Parameters
        ----------
        dataset: pm.Dataset
            ...

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

            rm_df = self.compute_reading_measures(
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

    def compute_reading_measures(
            self,
            fixations_df: pd.DataFrame,
            aoi_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Computes reading measures from fixation sequences.

        Parameters
        ----------
        fixations_df : pd.DataFrame
            DataFrame with fixation data, containing columns 'index', 'duration',
            'aoi', 'word_roi_str'.
        aoi_df : pd.DataFrame
            DataFrame with AOI data, containing columns 'word_index', 'word',
            and the AOIs of each word.

        Returns
        -------
        pd.DataFrame
            DataFrame with computed reading measures.
        """
        # Append an extra dummy fixation to have the next fixation for the actual last fixation.
        fixations_df = pd.concat(
            [
                fixations_df,
                pd.DataFrame(
                    [[0 for _ in range(len(fixations_df.columns))]],
                    columns=fixations_df.columns,
                ),
            ],
            ignore_index=True,
        )

        # Adjust AOI indices (fix off by one error).
        aoi_df['aoi'] = aoi_df['aoi'] - 1
        # Get original words of the text and their indices.
        text_aois = aoi_df['aoi'].tolist()
        text_strs = aoi_df['character'].tolist()

        # Initialize dictionary for reading measures per word.
        word_dict = {
            int(word_index): {
                'word': word,
                'word_index': word_index,
                'FFD': 0, 'SFD': 0, 'FD': 0, 'FPRT': 0, 'FRT': 0, 'TFT': 0, 'RRT': 0,
                'RPD_inc': 0, 'RPD_exc': 0, 'RBRT': 0, 'Fix': 0, 'FPF': 0, 'RR': 0,
                'FPReg': 0, 'TRC_out': 0, 'TRC_in': 0, 'SL_in': 0, 'SL_out': 0, 'TFC': 0,
            } for word_index, word in zip(text_aois, text_strs)
        }

        # Variables to track fixation progress.
        right_most_word, cur_fix_word_idx, next_fix_word_idx, next_fix_dur = -1, -1, -1, -1

        # Iterate over fixation data.
        for index, fixation in fixations_df.iterrows():
            try:
                aoi = int(fixation['aoi']) - 1
            except ValueError:
                continue

            # Update variables.
            last_fix_word_idx = cur_fix_word_idx
            cur_fix_word_idx = next_fix_word_idx
            cur_fix_dur = next_fix_dur
            if pd.isna(cur_fix_dur):
                continue

            next_fix_word_idx = aoi
            next_fix_dur = fixation['duration']

            if next_fix_dur == 0:
                next_fix_word_idx = cur_fix_word_idx

            if right_most_word < cur_fix_word_idx:
                right_most_word = cur_fix_word_idx

            if cur_fix_word_idx == -1:
                continue

            # Update reading measures for the current word.
            word_dict[cur_fix_word_idx]['TFT'] += int(cur_fix_dur)
            word_dict[cur_fix_word_idx]['TFC'] += 1
            if word_dict[cur_fix_word_idx]['FD'] == 0:
                word_dict[cur_fix_word_idx]['FD'] += int(cur_fix_dur)

            if right_most_word == cur_fix_word_idx:
                if word_dict[cur_fix_word_idx]['TRC_out'] == 0:
                    word_dict[cur_fix_word_idx]['FPRT'] += int(cur_fix_dur)
                    if last_fix_word_idx < cur_fix_word_idx:
                        word_dict[cur_fix_word_idx]['FFD'] += int(cur_fix_dur)
            else:
                word_dict[right_most_word]['RPD_exc'] += int(cur_fix_dur)

            if cur_fix_word_idx < last_fix_word_idx:
                word_dict[cur_fix_word_idx]['TRC_in'] += 1
            if cur_fix_word_idx > next_fix_word_idx:
                word_dict[cur_fix_word_idx]['TRC_out'] += 1
            if cur_fix_word_idx == right_most_word:
                word_dict[cur_fix_word_idx]['RBRT'] += int(cur_fix_dur)
            if (
                word_dict[cur_fix_word_idx]['FRT'] == 0 and
                (not next_fix_word_idx == cur_fix_word_idx or next_fix_dur == 0)
            ):
                word_dict[cur_fix_word_idx]['FRT'] = word_dict[cur_fix_word_idx]['TFT']
            if word_dict[cur_fix_word_idx]['SL_in'] == 0:
                word_dict[cur_fix_word_idx]['SL_in'] = cur_fix_word_idx - last_fix_word_idx
            if word_dict[cur_fix_word_idx]['SL_out'] == 0:
                word_dict[cur_fix_word_idx]['SL_out'] = next_fix_word_idx - cur_fix_word_idx

        # Finalize reading measures.
        for word_indices, word_rm in sorted(word_dict.items()):
            if word_rm['FFD'] == word_rm['FPRT']:
                word_rm['SFD'] = word_rm['FFD']
            word_rm['RRT'] = word_rm['TFT'] - word_rm['FPRT']
            word_rm['FPF'] = int(word_rm['FFD'] > 0)
            word_rm['RR'] = int(word_rm['RRT'] > 0)
            word_rm['FPReg'] = int(word_rm['RPD_exc'] > 0)
            word_rm['Fix'] = int(word_rm['TFT'] > 0)
            word_rm['RPD_inc'] = word_rm['RPD_exc'] + word_rm['RBRT']

            # Create or append to DataFrame.
            if word_indices == 0:
                rm_df = pd.DataFrame([word_rm])
            else:
                rm_df = pd.concat([rm_df, pd.DataFrame([word_rm])])

        return rm_df

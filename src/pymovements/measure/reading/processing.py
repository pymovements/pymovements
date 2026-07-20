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
"""Reading measure processing functions."""
from __future__ import annotations

import polars as pl


def compute_reading_measures(
        fixations_df: pl.DataFrame,
        aoi_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute reading measures from fixation sequences.

    Parameters
    ----------
    fixations_df : pl.DataFrame
        DataFrame with fixation data, containing columns 'index', 'duration',
        'aoi', 'word_roi_str'.
    aoi_df : pl.DataFrame
        DataFrame with AOI data, containing columns 'word_index', 'word',
        and the AOIs of each word.

    Returns
    -------
    pl.DataFrame
        DataFrame with computed reading measures.
    """
    # Append an extra dummy fixation to have the next fixation for the actual last fixation.
    dummy_fixation_dict: dict[str, list[int] | list[str]] = {}
    for col, dtype in fixations_df.schema.items():
        if dtype == pl.String:
            dummy_fixation_dict[col] = ['']
        else:
            dummy_fixation_dict[col] = [0]
    dummy_fixation = pl.DataFrame(
        dummy_fixation_dict,
        schema=fixations_df.schema,
    )
    fixations_df = pl.concat([fixations_df, dummy_fixation])

    # Adjust AOI indices (fix off by one error).
    aoi_df = aoi_df.with_columns(
        (pl.col('aoi') - 1).alias('aoi'),
    )

    # Get original words of the text and their indices.
    text_aois = aoi_df['aoi'].to_list()
    text_strs = aoi_df['character'].to_list()

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

    # Add a catch-all entry for the dummy fixation and invalid AOIs
    word_dict[-1] = {
        'word': None, 'word_index': -1,
        'FFD': 0, 'SFD': 0, 'FD': 0, 'FPRT': 0, 'FRT': 0, 'TFT': 0, 'RRT': 0,
        'RPD_inc': 0, 'RPD_exc': 0, 'RBRT': 0, 'Fix': 0, 'FPF': 0, 'RR': 0,
        'FPReg': 0, 'TRC_out': 0, 'TRC_in': 0, 'SL_in': 0, 'SL_out': 0, 'TFC': 0,
    }

    # Variables to track fixation progress.
    right_most_word, cur_fix_word_idx, next_fix_word_idx, next_fix_dur = -1, -1, -1, -1

    # Iterate over fixation data.
    for fixation in fixations_df.to_dicts():
        try:
            aoi = int(fixation['aoi']) - 1
            if aoi not in word_dict:
                continue
        except (ValueError, TypeError):
            continue

        # Update variables.
        last_fix_word_idx = cur_fix_word_idx
        cur_fix_word_idx = next_fix_word_idx
        cur_fix_dur = next_fix_dur
        if cur_fix_dur is None:
            continue

        next_fix_word_idx = aoi
        next_fix_dur = fixation['duration']

        if next_fix_dur == 0 and not next_fix_word_idx == -1:
            next_fix_word_idx = cur_fix_word_idx

        right_most_word = max(right_most_word, cur_fix_word_idx)

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
    rm_list = []
    for word_idx, word_rm in sorted(word_dict.items()):
        if word_idx == -1:
            continue
        if word_rm['FFD'] == word_rm['FPRT']:
            word_rm['SFD'] = word_rm['FFD']
        word_rm['RRT'] = word_rm['TFT'] - word_rm['FPRT']
        word_rm['FPF'] = int(word_rm['FFD'] > 0)
        word_rm['RR'] = int(word_rm['RRT'] > 0)
        word_rm['FPReg'] = int(word_rm['RPD_exc'] > 0)
        word_rm['Fix'] = int(word_rm['TFT'] > 0)
        word_rm['RPD_inc'] = word_rm['RPD_exc'] + word_rm['RBRT']

        rm_list.append(word_rm)

    return pl.DataFrame(rm_list)

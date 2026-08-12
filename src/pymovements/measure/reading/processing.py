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
        fixations: pl.DataFrame,
        aois: pl.DataFrame,
        *,
        word_index_column: str = 'word_idx',
        word_column: str = 'word',
) -> pl.DataFrame:
    """Compute reading measures from fixation sequences.

    This function expects fixations annotated with AOI data. See
    :py:meth:`~pymovements.Events.map_to_aois` for further details.

    Parameters
    ----------
    fixations : pl.DataFrame
        DataFrame with fixation data, containing the column specified by ``word_index_column``.
    aois : pl.DataFrame
        DataFrame with AOI data, containing the columns specified by ``word_index_column`` and
        ``word_column``.
    word_index_column : str
        Shared column name in ``fixations`` and ``aois`` that corresponds to the word index of the
        text.
        (default: ``'aoi'``)
    word_column : str
        Column in ``aois`` with the content within each AOI.
        (default: ``'word'``)

    Returns
    -------
    pl.DataFrame
        DataFrame with computed reading measures.
    """
    # Normalize one-based AOI indices while preserving zero-based inputs.
    index_offset = 0 if aois[word_index_column].eq(0).any() else 1

    # Append an extra dummy fixation to have the next fixation for the actual last fixation.
    dummy_fixation_dict: dict[str, list[int] | list[str]] = {}
    for col, dtype in fixations.schema.items():
        if dtype == pl.String:
            dummy_fixation_dict[col] = ['']
        elif col == word_index_column:
            dummy_fixation_dict[col] = [index_offset - 1]
        else:
            dummy_fixation_dict[col] = [0]
    dummy_fixation = pl.DataFrame(
        dummy_fixation_dict,
        schema=fixations.schema,
    )
    fixations = pl.concat([fixations, dummy_fixation])

    # Get the original words of the text and their normalized indices.
    word_indices = [
        int(word_index) - index_offset
        for word_index in aois[word_index_column].to_list()
    ]
    words = aois[word_column].to_list()

    # Initialize dictionary for reading measures per word.
    rm_dict = {
        word_index: {
            'word': word,
            'word_index': word_index,
            'FFD': 0, 'SFD': 0, 'FD': 0, 'FPRT': 0, 'FRT': 0, 'TFT': 0, 'RRT': 0,
            'RPD_inc': 0, 'RPD_exc': 0, 'RBRT': 0, 'Fix': 0, 'FPF': 0, 'RR': 0,
            'FPReg': 0, 'TRC_out': 0, 'TRC_in': 0, 'SL_in': 0, 'SL_out': 0, 'TFC': 0,
        } for word_index, word in zip(word_indices, words)
    }

    # Add a catch-all entry for the dummy fixation and invalid AOIs
    rm_dict[-1] = {
        'word': None, 'word_index': -1,
        'FFD': 0, 'SFD': 0, 'FD': 0, 'FPRT': 0, 'FRT': 0, 'TFT': 0, 'RRT': 0,
        'RPD_inc': 0, 'RPD_exc': 0, 'RBRT': 0, 'Fix': 0, 'FPF': 0, 'RR': 0,
        'FPReg': 0, 'TRC_out': 0, 'TRC_in': 0, 'SL_in': 0, 'SL_out': 0, 'TFC': 0,
    }

    # Variables to track fixation progress.
    right_most_word, cur_fix_word_idx, next_fix_word_idx, next_fix_dur = -1, -1, -1, -1

    # Iterate over fixation data.
    for fixation in fixations.to_dicts():
        try:
            aoi = int(fixation[word_index_column]) - index_offset
            if aoi not in rm_dict:
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
        rm_dict[cur_fix_word_idx]['TFT'] += int(cur_fix_dur)
        rm_dict[cur_fix_word_idx]['TFC'] += 1
        if rm_dict[cur_fix_word_idx]['FD'] == 0:
            rm_dict[cur_fix_word_idx]['FD'] += int(cur_fix_dur)

        if right_most_word == cur_fix_word_idx:
            if rm_dict[cur_fix_word_idx]['TRC_out'] == 0:
                rm_dict[cur_fix_word_idx]['FPRT'] += int(cur_fix_dur)
                if last_fix_word_idx < cur_fix_word_idx:
                    rm_dict[cur_fix_word_idx]['FFD'] += int(cur_fix_dur)
        else:
            rm_dict[right_most_word]['RPD_exc'] += int(cur_fix_dur)

        if cur_fix_word_idx < last_fix_word_idx:
            rm_dict[cur_fix_word_idx]['TRC_in'] += 1
        if cur_fix_word_idx > next_fix_word_idx:
            rm_dict[cur_fix_word_idx]['TRC_out'] += 1
        if cur_fix_word_idx == right_most_word:
            rm_dict[cur_fix_word_idx]['RBRT'] += int(cur_fix_dur)
        if (
            rm_dict[cur_fix_word_idx]['FRT'] == 0 and
            (not next_fix_word_idx == cur_fix_word_idx or next_fix_dur == 0)
        ):
            rm_dict[cur_fix_word_idx]['FRT'] = rm_dict[cur_fix_word_idx]['TFT']
        if rm_dict[cur_fix_word_idx]['SL_in'] == 0:
            rm_dict[cur_fix_word_idx]['SL_in'] = cur_fix_word_idx - last_fix_word_idx
        if rm_dict[cur_fix_word_idx]['SL_out'] == 0:
            rm_dict[cur_fix_word_idx]['SL_out'] = next_fix_word_idx - cur_fix_word_idx

    # Finalize reading measures.
    rm_list = []
    for aoi_key, aoi_rm in sorted(rm_dict.items()):
        if aoi_key == -1:
            continue
        if aoi_rm['FFD'] == aoi_rm['FPRT']:
            aoi_rm['SFD'] = aoi_rm['FFD']
        aoi_rm['RRT'] = aoi_rm['TFT'] - aoi_rm['FPRT']
        aoi_rm['FPF'] = int(aoi_rm['FFD'] > 0)
        aoi_rm['RR'] = int(aoi_rm['RRT'] > 0)
        aoi_rm['FPReg'] = int(aoi_rm['RPD_exc'] > 0)
        aoi_rm['Fix'] = int(aoi_rm['TFT'] > 0)
        aoi_rm['RPD_inc'] = aoi_rm['RPD_exc'] + aoi_rm['RBRT']

        rm_list.append(aoi_rm)

    return pl.DataFrame(rm_list)

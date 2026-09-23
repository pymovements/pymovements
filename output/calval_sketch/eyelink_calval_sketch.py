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

"""SKETCH: parse the full EyeLink calibration/validation content into flat frames.

This is exploratory code for #1731 and the MultiplEYE downstream needs. It is
deliberately NOT wired into ``Gaze``/``from_asc`` and not polished. It only
demonstrates *how* the ASC content could be laid out as pymovements frames.

Design constraints discovered:
  * polars ``write_csv`` refuses nested data, so per-point / array-valued data
    cannot live as list-of-struct columns. Everything here is flat scalars.
  * calibration has no numeric error, only a GOOD/FAILED usability flag.
  * validation has numeric avg/max errors + OFFSET, plus a GOOD/FAIR/POOR flag.
  * ABORTED validations have no scores at all.

Run:
    python output/calval_sketch/eyelink_calval_sketch.py <path-to-asc>
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import polars as pl

MSG_TIME = r'MSG\s+(?P<timestamp>\d+[.]?\d*)\s+'
NUM = r'[-+]?\d+(?:\.\d+)?'

CAL_HEADER = (
    r'>>>>>>>\s+CALIBRATION\s+\(HV(?P<num_points>\d+),(?P<type>[^)]+)\)\s+FOR\s+'
    r'(?P<tracked_eye>LEFT|RIGHT):'
)
CAL_GRADE = (
    MSG_TIME + r'!CAL\s+CALIBRATION\s+HV\d+\s+\S+\s+'
    r'(?P<tracked_eye>LEFT|RIGHT)\s+(?P<quality>GOOD|FAILED)'
)
CAL_POINT = (
    MSG_TIME + r'!CAL\s+(?P<target_x>' + NUM + r'),\s*(?P<target_y>' + NUM + r')\s+'
    r'(?P<measured_x>[-+]?\d+),\s*(?P<measured_y>[-+]?\d+)'
)
CAL_GAINS = (
    MSG_TIME + r'!CAL\s+Gains:\s+\w+:(?P<v1>' + NUM + r')\s+\w+:(?P<v2>' + NUM + r')'
    r'\s+\w+:(?P<v3>' + NUM + r')'
)
CAL_ARRAY_HEADER = (
    MSG_TIME + r'!CAL\s+(?P<name>eye check box|href cal range|Cal coeff|Corner correction)'
)
CAL_QUAD_CENTER = MSG_TIME + r'!CAL\s+Quadrant center:\s+centx,\s*centy\s*='
CAL_PRENORM = MSG_TIME + \
    r'!CAL\s+Prenormalize:\s+offx,\s*offy\s*=\s*(?P<x>' + NUM + r')\s+(?P<y>' + NUM + r')'
CAL_RES_CENTER = MSG_TIME + \
    r'!CAL\s+Resolution \(upd\) at screen center:\s+X=(?P<x>' + NUM + r'),\s*Y=(?P<y>' + NUM + r')'
CAL_GAIN_CHANGE = MSG_TIME + \
    r'!CAL\s+Gain Change Proportion:\s+X:\s*(?P<x>' + NUM + r')\s+Y:\s*(?P<y>' + NUM + r')'
CAL_GAIN_RATIO = MSG_TIME + r'!CAL\s+Gain Ratio \(Gy/Gx\)\s*=\s*(?P<v>' + NUM + r')'
CAL_BAD_RATIO = MSG_TIME + r'!CAL\s+Bad Y/X gain ratio:\s*(?P<v>' + NUM + r')'
CAL_CROSS_RATIO = MSG_TIME + \
    r'!CAL\s+Cross-Gain Ratios:\s*X=(?P<x>' + NUM + r'),\s*Y=(?P<y>' + NUM + r')'
CAL_PCR = MSG_TIME + \
    r'!CAL\s+PCR gain ratio\(x,y\)\s*=\s*(?P<x>' + NUM + r'),\s*(?P<y>' + NUM + r')'
CAL_CR = MSG_TIME + r'!CAL\s+CR gain match\(x,y\)\s*=\s*(?P<x>' + NUM + r'),\s*(?P<y>' + NUM + r')'
CAL_QUAD_FIXUP = MSG_TIME + \
    r'!CAL\s+Quadrant fixup\[(?P<i>\d+)\]\s*=\s*(?P<x>' + NUM + r'),(?P<y>' + NUM + r')'
CAL_SLIP = MSG_TIME + r'!CAL\s+Slip rotation correction\s+(?P<v>ON|OFF)'
CAL_WARNING = MSG_TIME + r'!CAL\s+(?P<msg>[A-Z][^:]*?(?:out of range|too large|diagonals))'

VAL_SUMMARY = (
    MSG_TIME + r'!CAL\s+VALIDATION\s+HV(?P<num_points>\d+)\s+\S+\s+(?P<tracked_eye>LEFT|RIGHT)\s+'
    r'(?P<quality>GOOD|FAIR|POOR|FAILED)\s+ERROR\s+'
    r'(?P<avg>' + NUM + r')\s+avg\.\s+(?P<max>' + NUM + r')\s+max\s+'
    r'OFFSET\s+(?P<offset>' + NUM + r')\s+deg\.\s+(?P<ox>' + NUM + r'),(?P<oy>' + NUM + r')\s+pix\.'
)
VAL_ABORTED = MSG_TIME + r'!CAL\s+VALIDATION\s+(?P<eye>[LR])\s+ABORTED'
VAL_POINT = (
    MSG_TIME + r'VALIDATE\s+(?P<mode>[LR]+)\s+(?:(?P<label>\d*POINT)\s+)?(?P<index>\d+)\s+'
    r'(?P<tracked_eye>LEFT|RIGHT)\s+at\s+(?P<tx>\d+),(?P<ty>\d+)\s+OFFSET\s+'
    r'(?P<offset>' + NUM + r')\s+deg\.\s+(?P<ox>' + NUM + r'),(?P<oy>' + NUM + r')\s+pix\.'
)
# TODO: DRIFTCORRECT <L|R|LR> <EYE|ABORTED> at x,y OFFSET d deg. x,y pix.
#       suggested follow-up: a dedicated ``drift_corrections`` frame.

# (name, n_rows, n_cols) for the calibration array blocks.
CAL_ARRAYS = {
    'eye check box': (1, 4),
    'href cal range': (1, 4),
    'Cal coeff': (2, 5),
    'Corner correction': (4, 2),
}


def _floats(line: str) -> list[float] | None:
    try:
        return [float(tok) for tok in line.replace(',', ' ').split()]
    except ValueError:
        return None


def parse_asc(path: str | Path) -> dict[str, pl.DataFrame]:
    lines = Path(path).read_text(encoding='utf-8', errors='ignore').splitlines()

    calibrations: list[dict[str, Any]] = []
    cal_points: list[dict[str, Any]] = []
    cal_params: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    val_points: list[dict[str, Any]] = []

    cal_timestamp: str | None = None
    cur_cal: dict[str, Any] | None = None
    pending: tuple[str, int, int, list[list[float]]] | None = None

    def finish_pending() -> None:
        nonlocal pending
        if pending is None or cur_cal is None:
            pending = None
            return
        name, _nrows, _ncols, rows = pending
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                cal_params.append({
                    'time': cur_cal['time'],
                    'eye': cur_cal['eye'],
                    'name': name,
                    'index': r * len(row) + c,
                    'value': value,
                })
        pending = None

    for raw in lines:
        line = raw.rstrip('\n')

        if _match(MSG_TIME + r'!CAL\s*$', line):
            cal_timestamp = _group(line, MSG_TIME)['timestamp']
            continue

        if m := _re(CAL_HEADER, line):
            finish_pending()
            cur_cal = {
                'time': cal_timestamp,
                'num_points': int(m['num_points']),
                'eye': m['tracked_eye'].lower(),
                'tracking_mode': m['type'],
                'quality': None,
                'prenormalize_offx_dva': None,
                'prenormalize_offy_dva': None,
                'gain_cx': None, 'gain_lx': None, 'gain_rx': None,
                'gain_cy': None, 'gain_ty': None, 'gain_by': None,
                'resolution_center_x': None, 'resolution_center_y': None,
                'gain_change_proportion_x': None, 'gain_change_proportion_y': None,
                'gain_ratio_gy_gx': None,
                'bad_yx_gain_ratio': None,
                'cross_gain_ratio_x': None, 'cross_gain_ratio_y': None,
                'pcr_gain_ratio_x': None, 'pcr_gain_ratio_y': None,
                'cr_gain_match_x': None, 'cr_gain_match_y': None,
                'slip_rotation_correction': None,
                'warnings': None,
            }
            calibrations.append(cur_cal)
            continue

        if cur_cal is not None:
            if m := _re(CAL_POINT, line):
                cal_points.append({
                    'time': m['timestamp'],
                    'eye': cur_cal['eye'],
                    'target_x_dva': float(m['target_x']),
                    'target_y_dva': float(m['target_y']),
                    'measured_x': float(m['measured_x']),
                    'measured_y': float(m['measured_y']),
                })
                continue
            if m := _re(CAL_GAINS, line):
                # two lines: cx/lx/rx then cy/ty/by. distinguish by first label.
                if line.split('Gains:')[1].strip().startswith('cx'):
                    cur_cal['gain_cx'], cur_cal['gain_lx'], cur_cal['gain_rx'] = (
                        float(m['v1']), float(m['v2']), float(m['v3']),
                    )
                else:
                    cur_cal['gain_cy'], cur_cal['gain_ty'], cur_cal['gain_by'] = (
                        float(m['v1']), float(m['v2']), float(m['v3']),
                    )
                continue
            if m := _re(CAL_ARRAY_HEADER, line):
                finish_pending()
                nrows, ncols = CAL_ARRAYS[m['name']]
                pending = (m['name'], nrows, ncols, [])
                continue
            if m := _re(CAL_QUAD_CENTER, line):
                pending = ('Quadrant center', 1, 2, [])
                continue
            if m := _re(CAL_PRENORM, line):
                cur_cal['prenormalize_offx_dva'] = float(m['x'])
                cur_cal['prenormalize_offy_dva'] = float(m['y'])
                continue
            if m := _re(CAL_RES_CENTER, line):
                cur_cal['resolution_center_x'] = float(m['x'])
                cur_cal['resolution_center_y'] = float(m['y'])
                continue
            if m := _re(CAL_GAIN_CHANGE, line):
                cur_cal['gain_change_proportion_x'] = float(m['x'])
                cur_cal['gain_change_proportion_y'] = float(m['y'])
                continue
            if m := _re(CAL_GAIN_RATIO, line):
                cur_cal['gain_ratio_gy_gx'] = float(m['v'])
                continue
            if m := _re(CAL_BAD_RATIO, line):
                cur_cal['bad_yx_gain_ratio'] = float(m['v'])
                continue
            if m := _re(CAL_CROSS_RATIO, line):
                cur_cal['cross_gain_ratio_x'] = float(m['x'])
                cur_cal['cross_gain_ratio_y'] = float(m['y'])
                continue
            if m := _re(CAL_PCR, line):
                cur_cal['pcr_gain_ratio_x'] = float(m['x'])
                cur_cal['pcr_gain_ratio_y'] = float(m['y'])
                continue
            if m := _re(CAL_CR, line):
                cur_cal['cr_gain_match_x'] = float(m['x'])
                cur_cal['cr_gain_match_y'] = float(m['y'])
                continue
            if m := _re(CAL_QUAD_FIXUP, line):
                for axis, value in (('x', m['x']), ('y', m['y'])):
                    cal_params.append({
                        'time': cur_cal['time'], 'eye': cur_cal['eye'],
                        'name': f'quadrant_fixup_{axis}', 'index': int(m['i']),
                        'value': float(value),
                    })
                continue
            if m := _re(CAL_SLIP, line):
                cur_cal['slip_rotation_correction'] = m['v'] == 'ON'
                continue
            if m := _re(CAL_WARNING, line):
                existing = cur_cal['warnings'] or ''
                cur_cal['warnings'] = (existing + '|' + m['msg']).strip('|')
                continue
            if m := _re(CAL_GRADE, line):
                cur_cal['quality'] = m['quality']
                finish_pending()
                cur_cal = None
                continue
            if pending is not None and not line.startswith('MSG'):
                values = _floats(line)
                if values is not None:
                    name, nrows, ncols, rows = pending
                    rows.append(values)
                    if len(rows) == nrows:
                        finish_pending()
                    continue

        if m := _re(VAL_SUMMARY, line):
            validations.append({
                'time': m['timestamp'],
                'num_points': int(m['num_points']),
                'eye': m['tracked_eye'].lower(),
                'quality': m['quality'],
                'accuracy_avg': float(m['avg']),
                'accuracy_max': float(m['max']),
                'offset_dva': float(m['offset']),
                'offset_x_pix': float(m['ox']),
                'offset_y_pix': float(m['oy']),
            })
            continue
        if m := _re(VAL_ABORTED, line):
            validations.append({
                'time': m['timestamp'],
                'num_points': None,
                'eye': 'left' if m['eye'] == 'L' else 'right',
                'quality': 'ABORTED',
                'accuracy_avg': None, 'accuracy_max': None,
                'offset_dva': None, 'offset_x_pix': None, 'offset_y_pix': None,
            })
            continue
        if m := _re(VAL_POINT, line):
            val_points.append({
                'time': m['timestamp'],
                'eye': m['tracked_eye'].lower(),
                'point_label': m['label'],
                'point_index': int(m['index']),
                'target_x_pix': int(m['tx']),
                'target_y_pix': int(m['ty']),
                'offset_dva': float(m['offset']),
                'offset_x_pix': float(m['ox']),
                'offset_y_pix': float(m['oy']),
            })
            continue

    return {
        'calibrations': _frame(calibrations),
        'calibration_points': _frame(cal_points),
        'calibration_parameters': _frame(cal_params),
        'validations': _frame(validations),
        'validation_points': _frame(val_points),
    }


def _frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    return pl.from_dicts(rows)


# --- tiny regex helpers (sketch only) ---------------------------------------------------

def _re(pattern: str, line: str) -> dict[str, str] | None:
    match = _re_compile(pattern).match(line)
    return match.groupdict() if match else None


def _match(pattern: str, line: str) -> bool:
    return _re_compile(pattern).match(line) is not None


def _group(line: str, pattern: str) -> dict[str, str]:
    match = _re_compile(pattern).match(line)
    assert match is not None
    return match.groupdict()


_CACHE: dict[str, Any] = {}


def _re_compile(pattern: str):
    import re
    if pattern not in _CACHE:
        _CACHE[pattern] = re.compile(pattern)
    return _CACHE[pattern]


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'tests/files/eyelink_monocular_example.asc'
    frames = parse_asc(target)
    for name, frame in frames.items():
        print(f'--- {name} ({frame.height} rows) ---')
        print(frame)
        print()

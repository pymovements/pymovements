# EyeLink calibration and validation content: what we have, what we drop, what downstream needs

Discussion note for maintainers. This is not a polished proposal, it is intended to
collect feedback before we commit to a schema. It relates to #1731 and to the
downstream MultiplEYE preprocessing work.

The runnable sketch referenced here is:
`output/calval_sketch/eyelink_calval_sketch.py`
Run it with `python output/calval_sketch/eyelink_calval_sketch.py <file.asc>`.

## 1. What exists on `main` today

`from_asc` builds two public frames.

`gaze.calibrations`:
- `time`, `num_points`, `eye`, `tracking_mode`

`gaze.validations`:
- `time`, `num_points`, `eye`, `accuracy_avg`, `accuracy_max`

Both are summary only. The raw ASC lines are recoverable only via `messages=True`, where
they stay unstructured text.

## 2. What the ASC actually contains

Based on a corpus wide scan of 209 ASC files (about 64 GB, MultiplEYE sessions) plus the
EyeLink 1000 Plus manual and SR Research support pages.

### Calibration block

- grade line: `!CAL CALIBRATION HV9 L LEFT GOOD` (values: `GOOD`, `FAILED`)
- header: `>>>>>>> CALIBRATION (HV9,P-CR) FOR LEFT:` gives `num_points`, `type`, `eye`
- point lines: `!CAL <target_x>, <target_y>  <measured_x>, <measured_y>` (one per point).
  `target_*` is in degrees of visual angle. `measured_*` is in EyeLink raw sensor units.
- model scalars: `Prenormalize offx/offy`, `Gains cx/lx/rx` and `cy/ty/by`,
  `Resolution (upd) at screen center X/Y`, `Gain Change Proportion X/Y`,
  `Gain Ratio (Gy/Gx)`, `PCR gain ratio x/y`, `CR gain match x/y`,
  `Cross-Gain Ratios X/Y`, `Bad Y/X gain ratio`, `Slip rotation correction ON/OFF`
- array diagnostics (values on following lines):
  `eye check box` (4 ints), `href cal range` (4 ints), `Cal coeff` (2x5 floats),
  `Quadrant center` (2 floats), `Corner correction` (4x2 floats),
  `Quadrant fixup[i]` (pairs)
- warning lines: `X gain out of range`, `Y gain out of range`,
  `X/Y Nonlinearity Proportion too large`, `WARNING: diagonals`

Important: calibration has no numeric error. `GOOD`/`FAILED` is a usability flag, not an
error threshold. SR Research rates calibration only as usable or not, and references the
validation for actual accuracy. So there is no number to derive or to add for calibration.

### Validation block

- summary line:
  `!CAL VALIDATION HV9 L LEFT GOOD ERROR 0.13 avg. 0.60 max OFFSET 0.04 deg. 0.7,1.1 pix.`
  gives `num_points`, `eye`, `quality`, `avg`, `max`, `offset_dva`, `offset_x_pix`,
  `offset_y_pix`
- point lines:
  `VALIDATE L POINT 0 LEFT at 661,490 OFFSET 0.06 deg. 1.5,1.2 pix.`
  gives point label, index, eye, target pixel position, offset in dva and pixels
- aborted line: `!CAL VALIDATION L ABORTED` (or `R`), no scores at all
- drift correction: `DRIFTCORRECT R RIGHT at x,y OFFSET d deg. x,y pix.`
  and `DRIFTCORRECT R ABORTED`

Validation quality has a numeric basis. SR Research system criteria:
- `GOOD`: average error below 1.0 deg and max error below 1.5 deg
- `FAIR`: average error between 1.0 and 1.5 deg or max error between 1.5 and 2.0 deg
- `POOR`: average error above 1.5 deg or max error above 2.0 deg

The average is weighted by target position (center weighted more), so it should be taken
from the file, not recomputed as a plain mean of the per point offsets.

Observed value sets in the corpus:
- calibration: `GOOD` dominant, `FAILED` rare
- validation: `GOOD`, `FAIR`, `POOR`, `ABORTED`

## 3. What pymovements currently drops

Calibration: everything except `time`, `num_points`, `eye`, `tracking_mode`, and now the
new `quality` flag on this branch. This includes all point data, all model diagnostics, and
warning reasons.

Validation: the `quality` flag, the summary `OFFSET`, and all per point data. In addition
there is a parsing bug: the score groups are `\d.\d\d`, so any multi-digit average or max
(for example `12.37 avg. 25.25 max`) makes the whole line fail to match and the validation
row is dropped entirely, including its quality label. That is a data loss bug, not just a
missing label. A real example was found in the corpus.

ABORTED validations produce no row at all today.

## 4. What we want for downstream

Minimum: the calibration and validation quality labels as normal frame columns, since
downstream currently re reads the whole ASC just to recover them. This is what the branch
already does.

Ideal: land everything the ASC provides into typed frames so downstream does not need to
re parse the file. Concretely, all of the missing fields listed in section 2.

## 5. Proposed frame layout for discussion

Constraint discovered: polars `write_csv` refuses nested data
(`ComputeError: CSV format does not support nested data`). So list of struct columns would
break `gaze.save(extension='csv')`. Downstream reads TSV. Therefore all new data is flat.

Proposed frames:

- `gaze.calibrations` (scalars): existing columns plus `quality`, `prenormalize_offx_dva`,
  `prenormalize_offy_dva`, `gain_cx/lx/rx`, `gain_cy/ty/by`, `resolution_center_x/y`,
  `gain_change_proportion_x/y`, `gain_ratio_gy_gx`, `bad_yx_gain_ratio`,
  `cross_gain_ratio_x/y`, `pcr_gain_ratio_x/y`, `cr_gain_match_x/y`,
  `slip_rotation_correction` (bool), `warnings` (joined string)
- `gaze.calibration_points` (flat): `time`, `eye`, `target_x_dva`, `target_y_dva`,
  `measured_x`, `measured_y`
- `gaze.calibration_parameters` (flat long): `time`, `eye`, `name`, `index`, `value`
  for `Cal coeff`, `Corner correction`, `eye check box`, `href cal range`,
  `Quadrant center`, `Quadrant fixup`
- `gaze.validations` (scalars): existing plus `quality`, `offset_dva`, `offset_x_pix`,
  `offset_y_pix`, and the widened score regex so multi-digit values parse
- `gaze.validation_points` (flat): `time`, `eye`, `point_label`, `point_index`,
  `target_x_pix`, `target_y_pix`, `offset_dva`, `offset_x_pix`, `offset_y_pix`

Joining points to their parent uses `(time, eye)`, which matches in the real data.

Drift corrections: suggest a later `gaze.drift_corrections` frame. Left as a TODO in the
sketch for now.

## 6. Open questions for feedback

1. Do we want all of this in `Gaze`, or only the labels in the near term and the rest
   behind an opt in flag to avoid carrying point data by default?
2. Is the flat long `calibration_parameters` frame acceptable, or do we prefer expanding
   the fixed arity arrays into scalar columns?
3. Naming and units. Should new columns carry a unit suffix such as `_dva`? Note that the
   existing validation columns `accuracy_avg` and `accuracy_max` already exist on `main`
   without a unit suffix, so renaming them would be a breaking change.
4. Do we want a public helper that maps validation average and max error to the SR
   `GOOD`/`FAIR`/`POOR` categories, with configurable cutoffs? Calibration has no numbers,
   so no equivalent there.
5. Should `ABORTED` validations be rows with null scores (they are in the sketch) or stay
   out of the frame?

## 7. Known bugs to fix regardless of the schema decision

- widen the validation score groups so multi-digit `avg` and `max` parse
- add `ABORTED` rows so validation counts are complete
- keep the existing `error` field values unchanged for backward compatibility

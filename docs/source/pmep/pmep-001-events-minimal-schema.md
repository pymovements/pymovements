# PMEP 1: Exact event durations with minimal schema: onset, duration, name

| | |
|---|---|
| **Status** | Draft |
| **Type** | Standards |
| **Author** | Daniel Krakowczyk |
| **Created** | 2026-09-16 |
| **Supersedes** | none |

Proposed in issue
[#1677](https://github.com/pymovements/pymovements/issues/1677). Blocks the BIDS events
save/load work ([#1563](https://github.com/pymovements/pymovements/issues/1563)) and closes
[#1083](https://github.com/pymovements/pymovements/issues/1083).

## TL;DR

- The minimal `Events` schema becomes `onset`, `duration`, `name`, in BIDS order.
- `offset` leaves the stored schema and becomes an on-demand measure with an `inclusive`
  parameter.
- Every duration pymovements produces today, detected or parsed, is one sampling interval
  short: `offset - onset` measures first sample to last. Durations become the time from event
  start to event end, `t_last - t_first + one sampling interval`, the duration EyeLink reports.
- The value change is an immediate breaking change in v0.29.0: all durations grow by one
  sampling interval. No compatibility phase, only a changelog entry and a versioned
  migration note.
- `offsets=` stays as an input alternative, but the convention must be stated via
  `offsets_inclusive`. The implicit default is removed in v0.34.0.
- `duration` is nullable: `0` means instantaneous, `null` means unavailable.

## What it looks like

Constructing the same four events, before and after. Frames built from `offsets=` or loaded
from a file are shown in their end state from v0.34.0 on. During the deprecation window they
keep the legacy `offset` column and the old column order (see Backwards compatibility).

```python
# before (v0.28): offsets are stored, duration is derived one sampling interval short
events = pymovements.Events(
    name=['fixation', 'saccade', 'fixation', 'blink'],
    onsets=[0, 121, 159, 301],
    offsets=[120, 158, 300, 380],
)
events.frame
# ┌──────────┬───────┬────────┬──────────┐
# │ name     ┆ onset ┆ offset ┆ duration │
# │ fixation ┆ 0     ┆ 120    ┆ 120      │
# │ saccade  ┆ 121   ┆ 158    ┆ 37       │
# │ fixation ┆ 159   ┆ 300    ┆ 141      │
# │ blink    ┆ 301   ┆ 380    ┆ 79       │
# └──────────┴───────┴────────┴──────────┘
```

```python
# after: durations are stored exactly, offset is on demand
events = pymovements.Events(
    name=['fixation', 'saccade', 'fixation', 'blink'],
    onsets=[0, 121, 159, 301],
    durations=[121, 38, 142, 80],
)
events.frame
# ┌───────┬──────────┬──────────┐
# │ onset ┆ duration ┆ name     │
# │ 0     ┆ 121      ┆ fixation │
# │ 121   ┆ 38       ┆ saccade  │
# │ 159   ┆ 142      ┆ fixation │
# │ 301   ┆ 80       ┆ blink    │
# └───────┴──────────┴──────────┘

# equivalently, from inclusive last-sample offsets (duration = offset - onset + Δ):
events = pymovements.Events(
    name=['fixation', 'saccade', 'fixation', 'blink'],
    onsets=[0, 121, 159, 301],
    offsets=[120, 158, 300, 380],
    offsets_inclusive=True,
    sampling_rate=1000,
)  # same frame as above
```

Loading an EyeLink file gives the same frame, with durations equal to the file's reported `DUR`
(v0.28 loads them one interval short). `parse_offset=True` keeps the parsed end timestamps:

```python
gaze = pymovements.gaze.from_asc('subject.asc', events=True, parse_offset=True)
gaze.events.frame
# ┌───────┬──────────┬──────────┬────────┐
# │ onset ┆ duration ┆ name     ┆ offset │
# │ 0     ┆ 121      ┆ fixation ┆ 120    │
# │ 121   ┆ 38       ┆ saccade  ┆ 158    │
# │ 159   ┆ 142      ┆ fixation ┆ 300    │
# │ 301   ┆ 80       ┆ blink    ┆ 380    │
# └───────┴──────────┴──────────┴────────┘
```

## Resulting signatures

The `Events` constructor gains `durations=` and the offset-convention parameters:

```python
Events(
    data: polars.DataFrame | None = None,
    *,
    name: str | list[str] | None = None,
    onsets: list[int | float] | np.ndarray | None = None,
    durations: list[int | float] | np.ndarray | None = None,   # new
    offsets: list[int | float] | np.ndarray | None = None,     # permanent alternative to durations=
    offsets_inclusive: bool | None = None,                     # None deprecated, raises v0.34.0
    sampling_rate: float | None = None,                        # needed if no rate resolves
    trials: ... = None,
    trial_columns: ... = None,
    time_unit: str | None = None,
)
```

The `offset` measure joins the event-measure registry:

```python
offset(inclusive: bool = True, sampling_rate: float | None = None) -> polars.Expr
# inclusive=True  (default): onset + duration - sampling_interval, the last-sample timestamp
# inclusive=False:           onset + duration, one past the end, the next onset for adjacent events
```

Vendor loaders that report an offset gain a retention parameter:

```python
pymovements.gaze.from_asc(file, *, parse_offset: bool | None = None, ...)
# parse_offset=True:  also store the parser's reported offset as an additional column
# parse_offset=False: the offset is not stored
# None default: behaves as True until v0.34.0 (DeprecationWarning), then as False
# explicit True/False are permanent
```

No on-disk format is defined. Saved event files mirror the frame schema.

## Motivation

`Events.duration` is derived as `offset - onset`. All producers store the offset as the
timestamp of the event's last sample (inclusive): the detection algorithms take
`timesteps[candidate_indices[-1]]`, and the EyeLink parser stores the `EFIX`/`ESACC`/`EBLINK`
end timestamps verbatim. The derived duration is therefore one sampling interval short of the
time the event spans. The first fixation of the running example, at 1000 Hz (`Δ = 1 ms`):

```text
onset  = 0 ms          timestamp of the first sample
offset = 120 ms        timestamp of the last sample
                       the fixation covers 121 samples, at 0, 1, 2, ..., 120 ms

derived duration = offset - onset      = 120 - 0     = 120 ms   ← one sample short
actual  duration = offset - onset + Δ  = 120 - 0 + 1 = 121 ms   ← EyeLink's reported DUR
```

The shortfall is exactly one interval `Δ`, so the absolute error scales with the sampling rate:

| sampling rate | interval `Δ` | duration error |
|---|---|---|
| 500 Hz  | 2 ms   | −2 ms   |
| 1000 Hz | 1 ms   | −1 ms   |
| 2000 Hz | 0.5 ms | −0.5 ms |

The EyeLink parser captures the reported `DUR` in a regex group and then discards it.

The bias has practical consequences:

- Every measure built on durations underestimates how long events last, and aggregates lose
  `n_events * Δ`. `events2timeratio` adds one interval to every event duration to patch this
  back, estimated from the samples or taken from its `sampling_rate` parameter.
- The `fill` detector already disagrees about the convention: it treats stored offsets as
  exclusive, so each event's last sample leaks into the unclassified events.
- BIDS export (#1563) would publish the short durations into shared scientific data.

## Specification

**Minimal schema.** The minimal `Events` schema is `onset`, `duration`, `name`, in BIDS order,
across the whole codebase. `offset` leaves the stored schema. The schema covers all
discrete-time events, including point events and events of unknown duration. Which event kinds
belong in `Events` rather than `Gaze.messages` is out of scope.

**Duration definition.** Duration is the time from the start of the event to its end:

- Events built from samples (detection algorithms, vendor parsers): `t_last - t_first + Δ`, where
  `Δ` is the nominal interval of the sampling rate in effect, equal to `n_samples * Δ` for
  gap-free events. This holds also for events spanning data gaps: data loss inside an event is
  a data-quality measure, not something duration encodes. This matches EyeLink's reported
  `DUR`.
- Events with exact start and end times, not tied to samples (future producers such as
  recording start/stop, calibrations or stimulus presentations): `end - start`, with no `+ Δ`.
  Their timestamps are the event boundaries, not sample positions, so no quantization
  correction applies.

**Nullability.** `duration` is nullable, with BIDS semantics: `0` means an instantaneous point
event, `null` means the duration is unavailable. Single-sample events get `Δ`, never `0` (see
the quantization note in Rationale). Duration aggregations skip `null` rows.

**Sample selection.** The half-open interval `[onset, onset + duration)` selects an event's
samples. `onset + duration` equals the next event's onset wherever the next event starts on
the immediately following sample, so adjacent events tile the timeline and share no boundary
sample. A `null` or `0` duration selects no samples. Sample-level operations (AOI mapping,
segmentation, `nullify_event_samples`, the `fill` detector's event mask) use this selection and
need no sampling interval.

**Producers compute duration at the source**, where the sampling information lives. The
EyeLink parser takes the reported duration verbatim (the currently discarded `duration_ms`
group). Detection algorithms compute `t_last - t_first + Δ` from their `timesteps` and construct
events via `durations=`. Producers need no `offsets_inclusive` parameter of their own: the
convention question only exists for externally supplied offsets.

**Retaining parsed offsets.** Loaders whose format reports offsets directly (EyeLink's end
timestamps) gain `parse_offset` to keep them as an additional column. The offset measure
reconstructs the same value for sample-built events, but the retained column guards the case
where a vendor's end timestamp and `DUR` disagree. The canonical schema stays
`onset`/`duration`/`name`.

**The offset measure.** `offset` becomes an on-demand event measure with an `inclusive`
parameter (default `True`, reproducing today's stored offsets). `inclusive=True` requires a
sampling rate, resolved in this order: explicit argument, the frame's `sampling_rate` column
where present, the frame's sampling rate, the experiment default. `inclusive=False` needs no
rate. The measure computes uniformly over all rows, since the frame does not record whether an
event was built from samples. For events with exact start and end times, `inclusive=False`
returns the end timestamp. `inclusive=True` subtracts `Δ` from a boundary that is not a sample
position, so callers holding such events use `inclusive=False`.

**Frame granularity.** An events frame is produced per recording and carries one sampling
rate. Durations are computed at the producer, where that rate is known, so they stay exact
however frames are combined later. Mixed rates arise only through concatenation, which lifts
the sampling rate from frame metadata into a per-event `sampling_rate` column for the offset
measure. This also matches BIDS, which ties one events file to one recording.

**Measure plumbing.** `compute_event_properties` already accepts `(name, {kwargs})`. The
internal `EventProcessor` learns to pass the kwargs through to event measures, a non-breaking
internal extension.

**Remaining offset consumers** move to onset/duration arithmetic:

- `Events.merge_subsequent_close_events`: the gap becomes
  `onset - (previous onset + previous duration)`, the merged duration
  `last onset + last duration - first onset`.
- `compute_event_properties` joins on `['name', 'onset', 'duration']` with `nulls_equal=True`,
  so events with `null` duration keep their rows, and selects samples by the half-open
  interval.
- `events2segmentation` and `segmentation2events` use the half-open selection.
- `measure_events_ratio` and `events2timeratio` keep merging overlapping intervals (#1713) on
  the new schema and gain a `duration_column` parameter in place of `offset_column`. They stop
  adding one interval to event durations. `sampling_rate` stays, since the total time range
  still spans `t_last - t_first + Δ` over the samples. Events with `null` duration contribute
  nothing.
- The `fill` detector's event mask uses the half-open selection, fixing its last-sample leak.

**Duration thresholds.** `minimum_duration` compares against the new duration values. The I-VT,
I-HMM, fill, microsaccade and blink detectors currently compare `t_last - t_first` against the
threshold and gain the `+ Δ`, so events that previously missed the threshold by exactly one
interval now pass. The blink detector's `maximum_duration` flips the other way: events that
previously passed by exactly one interval now fail. Subtracting one interval from
`minimum_duration` and adding one to `maximum_duration` restores the previous event sets. I-DT
already converts the threshold to a sample count, which is the same convention, so its event
sets do not change.

## Rationale

**Why add one sampling interval (quantization note).** No convention recovers the true
continuous-time duration of an individual event: the true start lies in `(t_first - Δ, t_first]`,
the true end in `[t_last, t_last + Δ)`. Start and end are each known to within one interval, and
duration is their difference, so both errors accumulate: the true duration lies in a window of
width `2Δ`, half the precision of onset and offset, under any convention. The two candidates sit
differently in that window: `offset - onset` is its floor, biased by `-Δ`, while
`t_last - t_first + Δ` is its center, unbiased. For the first fixation of the running example
the true duration lies in `[120, 122)` ms. The stored `121` is the center of that window and
EyeLink's reported `DUR`. Today's `120` is its edge, not a more precise value, only a biased one.
Three criteria pick the center independently of bias: additivity (durations sum to recording
length and adjacent events tile), the single-sample event (duration `Δ` instead of a degenerate
`0`), and agreement with the vendor's reported values.

**Why store duration and derive offset.** Duration is what analyses consume and what vendors
report. Offset depends on a convention (inclusive last sample or one past the end). Storing the
convention-free value and answering the convention question in one place, the measure, removes
the ambiguity that already produced the disagreeing `fill` detector. EyeLink's reported `DUR`
becomes the stored value, so file and frame can no longer contradict each other.

**Why the value change is immediate.** A deprecation window keeps an old and a new API shape
working side by side. A single `duration` column cannot hold both the old and the new number,
so deferring only ships a known-wrong value for five more releases. Bundling it with the
`polars.Duration` change (#1637) in v0.29.0 breaks the events schema once instead of twice and
keeps #1563 from publishing biased durations. The silent numeric shift is carried by the
changelog entry and the migration note.

**Alternatives rejected.**

- *Keep the offset stored and document the convention.* Keeps the bias patched per consumer
  rather than fixed at the source, and carries the redundant column into every save file.
- *Store exclusive offsets instead.* Makes `offset - onset` exact, but stores a value no
  vendor reports, breaks verbatim EyeLink round-trips, and leaves the inclusivity question in
  every consumer that compares against sample timestamps.
- *Keep durations non-nullable and let the BIDS events work add nullability.* Breaks the same
  schema twice, and #1563 requires `null` durations for BIDS `n/a` round-trips.

## Backwards compatibility

**Immediate breaking change (v0.29.0).** All detected and parsed durations grow by one sampling
interval. Signatures and, during the deprecation window, the frame shape stay unchanged.

**Offsets stay as input, with an explicit convention.** `offsets=` remains a permanent
alternative to `durations=`. The offset is converted to a duration at construction and never
stored. `offsets_inclusive` states the convention of the supplied offsets:

- `True`: inclusive last-sample timestamps, `duration = offset - onset + Δ`. Requires a
  sampling rate, resolved as for the offset measure.
- `False`: one past the end, `duration = offset - onset`. Needs no rate.
- `None` (default): interpreted as inclusive, with a `DeprecationWarning`. Raises if no
  sampling rate resolves.

**Deprecated v0.29.0, removed v0.34.0:**

- `offsets_inclusive=None`, as above.
- Loading legacy event files, which reuses `offsets_inclusive` through the constructor's
  `data` path: these files store `offset` and a derived `duration`, detectable as
  `duration == offset - onset` on all rows. With `None`, stored durations load untouched and a
  one-time warning points to the migration note. With `True` plus a sampling rate, durations
  are recomputed. With `False`, they are accepted as exact.
- The legacy `offset` column: detection algorithms and the `offsets=`/legacy-file paths keep
  materializing it, so `frame['offset']` and files saved during the window stay compatible.
  Events constructed from `durations=` alone already have the new shape.
- `parse_offset=None` on offset-reporting loaders, as in Resulting signatures.
- Frame column order: unchanged during the window (`[grouping columns,] name, onset, offset,
  ..., duration`), then flipped once to `onset`, `duration`, `name`, grouping and additional
  columns, the order #1563 needs.
- `offset_column` on `Gaze.measure_events_ratio` and `events2timeratio`: refers to a column
  that leaves the schema. `duration_column` replaces it.

**Migration note (versioned).** Documents the value change, the `minimum_duration` and
`maximum_duration` threshold adjustments, the legacy-file warning and opt-in correction, and
the v0.34.0 shape flip.

## Implementation

- [ ] `onset`/`duration`/`name` schema, `durations=` constructor path and `offsets_inclusive`
- [ ] `offset` event measure with `inclusive` and sampling-rate resolution
- [ ] concatenation lifts `sampling_rate` to a per-event column
- [ ] `EventProcessor` passes measure kwargs through
- [ ] EyeLink durations equal the file's `DUR`, tested on the ASC fixtures
- [ ] detected durations equal `t_last - t_first + Δ`
- [ ] `minimum_duration` compares against the new durations in I-VT, I-HMM, fill,
      microsaccades and blink, `maximum_duration` in blink
- [ ] `parse_offset` on offset-reporting loaders
- [ ] nullable durations
- [ ] offset consumers reworked: `merge_subsequent_close_events`, `compute_event_properties`,
      `events2segmentation`/`segmentation2events`, `measure_events_ratio`/`events2timeratio`
      with `duration_column`, `fill`
- [ ] legacy-file loading with detection warning and opt-in correction
- [ ] legacy `offset` column during the window, single shape flip at v0.34.0
- [ ] changelog entry and versioned migration note

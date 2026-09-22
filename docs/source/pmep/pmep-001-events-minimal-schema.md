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

Constructing the same four events, before and after:

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

Loading via `from_asc` reflects the same schema and the value change. Today (v0.28) it derives
durations one sampling interval short of the file's reported `DUR`. In the end state they are
exact, `offset` is dropped by default, and `parse_offset=True` retains the parser's reported
offset as a column:

```python
gaze = pymovements.gaze.from_asc('subject.asc', events=True)  # v0.28, 1000 Hz
gaze.events.frame
# ┌──────────┬───────┬────────┬──────────┐
# │ name     ┆ onset ┆ offset ┆ duration │
# │ fixation ┆ 0     ┆ 120    ┆ 120      │   file reports DUR 121
# │ saccade  ┆ 121   ┆ 158    ┆ 37       │   file reports DUR 38
# │ fixation ┆ 159   ┆ 300    ┆ 141      │   file reports DUR 142
# │ blink    ┆ 301   ┆ 380    ┆ 79       │   file reports DUR 80
# └──────────┴───────┴────────┴──────────┘

gaze = pymovements.gaze.from_asc('subject.asc', events=True, parse_offset=True)  # end state
gaze.events.frame
# ┌───────┬──────────┬──────────┬────────┐
# │ onset ┆ duration ┆ name     ┆ offset │
# │ 0     ┆ 121      ┆ fixation ┆ 120    │
# │ 121   ┆ 38       ┆ saccade  ┆ 158    │
# │ 159   ┆ 142      ┆ fixation ┆ 300    │
# │ 301   ┆ 80       ┆ blink    ┆ 380    │
# └───────┴──────────┴──────────┴────────┘
```

Without `parse_offset`, `offset` stays recoverable on demand via the `offset` measure (see
Resulting signatures): `inclusive=True` returns `onset + duration - Δ`, the last-sample
timestamps identical to v0.28's stored `offset`. `inclusive=False` returns `onset + duration`,
one past the end, so each offset equals the next event's onset.

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
# inclusive=False:           onset + duration, context-free
```

`measure_events_ratio` computes from the stored durations directly. Its `sampling_rate`
correction is deprecated (removal v0.34.0). Adjustments to its column parameters are settled
in the implementing issues.

Vendor loaders that report an offset gain a retention parameter:

```python
pymovements.gaze.from_asc(file, *, parse_offset: bool | None = None, ...)
# parse_offset=True:  also store the parser's reported offset as an additional column
# parse_offset=False: the offset is not stored
# None default: behaves as True during the deprecation window (DeprecationWarning),
#               flips to False at v0.34.0
```

No on-disk format is defined. Saved event files mirror the frame schema.

## Motivation

`Events.duration` is derived as `offset - onset`. All producers store the offset as the
timestamp of the event's last sample (inclusive): the detection algorithms take
`timesteps[candidate_indices[-1]]`, the EyeLink parser stores the `EFIX`/`ESACC`/`EBLINK` end
timestamps verbatim, and those coincide with the last sample. The derived duration is therefore
systematically one sampling interval short of the time the event actually spans. Take the first
fixation of the running example, sampled at 1000 Hz (interval `Δ = 1 ms`):

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

EyeLink reports the actual duration itself: its `DUR` field equals
`offset - onset + sampling_interval`. The parser captures that `DUR` value in a regex group and
then discards it, and `Events` re-derives the short value instead.

The bias has practical consequences:

- Every published measure built on durations underestimates how long events last, and
  aggregates lose `n_events * sampling_interval` per aggregate. `Gaze.measure_events_ratio`
  already carries an optional `sampling_rate` parameter solely to patch this back.
- One consumer already disagrees about the convention: the `fill` detector treats stored
  offsets as exclusive, applying `< offset` to inclusive last-sample timestamps, so each
  event's last sample leaks into the unclassified events.
- Any serialization boundary inherits the bias. BIDS export (#1563, blocked by this proposal)
  would publish the short durations into shared scientific data.

## Specification

**Minimal schema.** The minimal `Events` schema is `onset`, `duration`, `name`, in BIDS order,
across the whole codebase. `offset` leaves the stored schema.

**Scope.** The schema is defined for all discrete-time events, including point events and
events whose duration is unknown (see nullability below). Which event kinds belong in `Events`
as opposed to `Gaze.messages`, and any separation of event slots, is out of scope for this
proposal.

**Duration definition.** Duration is the time from the start of the event to its end:

- Events built from samples (detection algorithms, vendor parsers): `t_last - t_first + Δ`, where
  `Δ` is the nominal interval of the sampling rate in effect, equal to `n_samples * Δ` for
  gap-free events. This holds also for events spanning data gaps: data loss inside an event is
  a data-quality measure, not something duration encodes. This matches EyeLink's reported
  `DUR`.
- Events with exact start and end times, not tied to samples (future producers: recording
  start/stop, calibrations, validations, messages, stimulus presentations): `end - start`,
  with no `+ Δ`. Their timestamps are not sample positions, so no quantization correction
  applies.

**Quantization note.** No convention recovers the true continuous-time duration of an
individual event: the true start lies in `(t_first - Δ, t_first]`, the true end in
`[t_last, t_last + Δ)`. Start and end are each known to within one interval, and duration is
their difference, so both errors accumulate: the true duration lies in a window of width `2Δ`,
half the precision of onset and offset, under any convention. The two candidates sit
differently in that window: `offset - onset` is its floor (biased, expected error `-Δ`), while
`t_last - t_first + Δ` is its center (unbiased). For the first fixation of the running example
the true duration lies in `[120, 122)` ms. The stored `121` is the center of that window and
EyeLink's reported `DUR`, today's `120` its edge, not a more precise value, only a biased one.
Three criteria pick the center independently of bias: additivity (durations sum to recording
length and adjacent events tile), the single-sample event (duration `Δ` instead of a degenerate
`0`), and agreement with the vendor's reported values.

**Nullability.** `duration` is nullable, with BIDS-consistent semantics: `0` means an
instantaneous point event (button press, trigger, stimulus onset), `null` means the duration is
unavailable. Events built from a single sample get `Δ` per the quantization note, never
`0`. Duration aggregations skip `null` rows. Sample selection over a `null` or `0` duration
selects no samples.

**Sample selection and tiling.** The half-open interval `[onset, onset + duration)` selects an
event's samples: `onset <= t < onset + duration`. With the center convention these intervals
tile the timeline exactly: `onset + duration` equals the next event's onset wherever the next
event starts on the immediately following sample, and adjacent events share no boundary
sample. Sample-level operations (AOI mapping, segmentation, `nullify_event_samples`, the
`fill` detector's event mask) use this selection and need no sampling interval.

**Producers compute duration at the source**, where the sampling information lives. The
EyeLink parser takes the reported duration verbatim (the currently discarded `duration_ms`
group). Detection algorithms compute `t_last - t_first + Δ` from the `timesteps` array they
already hold and construct events via `durations=` directly. Producers need no
`offsets_inclusive` parameter of their own: the convention question only exists for externally
supplied offsets. (The legacy `offset` column during the deprecation window is covered in
Backwards compatibility.)

**Retaining parsed offsets.** Where a parser reports the offset directly (EyeLink's
`EFIX`/`ESACC`/`EBLINK` end timestamps), the loader gains a `parse_offset` parameter to keep
those parsed offsets as an additional column, so the value the file states is never silently
discarded and dropping the stored `offset` is not a downgrade for vendor-parsed events. For
sample-built events the offset measure (`inclusive=True`) reconstructs the same last-sample
timestamp, so the retained column normally duplicates what the measure reconstructs. It guards
the case where a vendor's reported end timestamp and its reported `DUR` are not exactly
consistent, which a derived value cannot recover. `parse_offset` also carries the loader side
of the deprecation window: the `None` default behaves as `True` during the window, so loaded
frames keep their `offset` column, and flips to `False` at v0.34.0 (see Backwards
compatibility). Explicit `True`/`False` are permanent. The canonical schema stays
`onset`/`duration`/`name`. The retained column is additional and not required for round-trips.

**The offset measure.** `offset` becomes an on-demand event measure with an `inclusive`
parameter (default `True`, reproducing today's stored offsets and EyeLink's printed end
timestamps). `inclusive=True` requires a sampling rate, while `inclusive=False` is
context-free. The resolution order is: explicit argument, then the frame's `sampling_rate`
column where one is present (concatenated frames, see frame granularity below), then the
frame's sampling rate, then the experiment default. `offset(inclusive=True)` is defined only
for events built from samples. Behavior for events with exact start and end times is specified
when producers for them land.

**Frame granularity.** An events frame is produced per recording and carries one sampling
rate. Durations are computed at the producer, where that rate is known, so they stay exact
regardless of how frames are later combined. Mixed rates arise only through concatenation,
which lifts the sampling rate from frame metadata into a per-event `sampling_rate` column
following the metadata-to-columns concatenation pattern, and the offset measure reads the
column where present and the frame's rate otherwise. Per-recording frames are also the shape
BIDS ties events files to, so the export in #1563 maps one frame to one events file.

**Measure plumbing.** `compute_event_properties` already accepts `(name, {kwargs})` at its
public signature. The internal `EventProcessor` currently instantiates event measures with zero
arguments and learns to pass the kwargs through, matching the samples-measure path. This is a
non-breaking internal extension.

**Remaining `offset` consumers** move to onset/duration arithmetic:

- `Events.merge`: the gap becomes `onset - (previous onset + previous duration)` and the
  merged duration becomes `last onset + last duration - first onset`.
- The `compute_event_properties` join key becomes `['name', 'onset', 'duration']` (unique, since
  same-name events from one detector cannot overlap). Sample selection uses the half-open
  interval.
- `events2segmentation` / `segmentation2events` use the half-open selection.
- `measure_events_ratio` merges overlapping half-open intervals of matching events and sums
  the merged durations, so each time point is counted once (the behavior introduced in #1713,
  restated on the new schema). Its `sampling_rate` parameter, which existed only to patch the
  duration bias, is deprecated. Events with `null` duration contribute nothing to the ratio.
- The `fill` detector's event mask uses the half-open selection, which also fixes its current
  exclusive-offset masking (each event's last sample leaks into the unclassified events).

**`minimum_duration`** compares against the new duration values, so the parameter means
what it says. Borderline events that previously failed the threshold by exactly one sampling
interval start passing. This is part of the same deliberate value change. The migration note
states that subtracting one interval from the threshold restores the previous event sets.

## Rationale

**Why store duration and derive offset, not the reverse.** Duration is what analyses consume
and what vendors report. Offset is convention-dependent (inclusive last sample vs one past the
end). Storing the convention-free value and answering the convention question in one place
(the measure) removes the ambiguity that already produced a disagreeing consumer (the
`fill` detector). The `duration_ms` value EyeLink prints becomes the value pymovements stores,
so file and frame can no longer contradict each other.

**Why the value change is immediate, not deprecated.** A deprecation window softens changes to
an API *shape* by keeping the old and new forms working side by side. A value change has no
shape to preserve: a single `duration` column cannot hand back both the old (short) and the new
(exact) number at once. Deferring therefore buys no gentle transition, it only ships a value
known to be wrong for five more releases, and there is nothing for callers to migrate *to*
except the corrected value. Bundling the change with the `polars.Duration` time-column change
(#1637) in v0.29.0 means the events schema breaks once at release level rather than twice, and
it keeps #1563 from publishing biased durations into shared BIDS datasets in the interim. The
one residual cost, a silent numeric shift with no error at the call site, is unavoidable for
any value correction. It is carried by the changelog entry and the versioned migration note,
including the `minimum_duration` threshold recipe.

**Alternatives rejected.**

- *Keep `offset` stored and document the convention.* Leaves two derived quantities
  (`duration`, and offsets under the other convention) and keeps the bias patched per consumer
  rather than fixed at the source. The redundant column also carries the conflict back into
  every save file.
- *Store exclusive offsets instead.* Makes `offset - onset` exact, but stores a value no
  vendor reports, breaks verbatim round-trips with EyeLink end timestamps, and still leaves
  the inclusivity question in every consumer that compares against sample timestamps.
- *Declare `duration` non-nullable and let the BIDS events work (#1563) add nullability.* The
  same schema would break twice, and #1563 states outright that it requires `null` durations
  for BIDS `n/a` round-trips.
- *Postpone the value change to the end of the deprecation window.* Would ship known-biased
  numbers for five more releases and force #1563 to publish them into BIDS datasets meanwhile.

## Backwards compatibility

**Immediate, deliberate breaking change (v0.29.0).** All detected and parsed duration values
grow by one sampling interval. Detection signatures and, during the window, the frame shape
stay unchanged. Only the numbers move. This corrects a biased value, so it gets no
compatibility phase: changelog entry plus a versioned migration note. It ships in the same
release as the `polars.Duration` time-column change (#1637), so the events schema breaks once
at release level.

**`offsets=` stays, but the convention must be stated.** `offsets=` remains a supported
alternative
to `durations=`: the offset is converted to a duration at construction and never stored, so the
frame still holds only `onset`/`duration`/`name`. The `offsets_inclusive` parameter states the
convention of the supplied offsets:

- `True`: inclusive last-sample timestamps. The conversion `duration = offset - onset + Δ`
  requires a resolvable `sampling_rate` (explicit argument, then the recording rate, then the
  experiment default).
- `False`: one past the end. The conversion `duration = offset - onset` needs no rate.
- `None` (default): interpreted as inclusive (the common case, matching pymovements' own
  producers and EyeLink), with a `DeprecationWarning` asking the caller to state the convention
  explicitly. If no sampling rate resolves, construction raises rather than silently keeping
  short values. This implicit default is removed in v0.34.0. From then on `offsets=` requires
  an explicit `offsets_inclusive`.

**Deprecated with the standard window (deprecated v0.29.0, removed v0.34.0):**

- The implicit offset convention (`offsets_inclusive=None`), as above.
- Loading legacy event files reuses `offsets_inclusive` through the constructor's `data` path.
  Legacy files store both `offset` and `duration` (`duration` is serialized by default), so the
  legacy pattern is detectable: `duration == offset - onset` on all rows means derived and
  short. With `None`, stored durations load untouched and the detected legacy pattern raises a
  one-time warning pointing to the migration note. Values never change behind the user's back.
  With `True` plus a sampling rate, durations are recomputed as the opt-in correction. With
  `False`, stored durations are accepted as exact.
- The legacy `offset` column: detection algorithms and the `offsets=`/legacy-file paths keep
  materializing it as an ordinary column (inclusive last sample), so `frame['offset']` access
  and files saved during the window stay compatible. Events constructed from `durations=`
  alone cannot carry it and already have the new shape.
- The `parse_offset=None` default on offset-reporting loaders: during the window it behaves
  as `True` (loaded frames keep their `offset` column), with a `DeprecationWarning` asking for
  an explicit choice. At v0.34.0 it flips to `False`, together with the shape flip. Explicit
  `True`/`False` are permanent.
- Frame column set and ordering: unchanged during the window
  (`[grouping columns,] name, onset, offset, ..., duration`). The shape flips once, at
  removal, directly to the BIDS ordering: `onset`, `duration`, `name`, then grouping and
  additional columns. This is the ordering #1563 needs, so the shape never flips again.
- `Gaze.measure_events_ratio(sampling_rate=...)`: redundant once durations are exact.

**Migration note (versioned).** Documents the value change, the threshold adjustment for
`minimum_duration`, the legacy-file warning and the opt-in correction recipe, and the v0.34.0
shape flip.

## Implementation

- [ ] minimal schema switched to `onset`/`duration`/`name`, with `offset` as an on-demand
      measure carrying an `inclusive` parameter (default `True`) and `sampling_rate` as tier 1
      of the resolution order
- [ ] events concatenation lifts `sampling_rate` from frame metadata to a per-event column,
      and the offset measure reads the column where present, the frame's rate otherwise
- [ ] `EventProcessor` passes `(name, {kwargs})` through to event measures
- [ ] `parse_offset` on offset-reporting loaders (e.g. `from_asc`), storing the parser's
      reported offset as an additional column, with the `None` default behaving as `True`
      during the window (`DeprecationWarning`) and flipping to `False` at v0.34.0
- [ ] EyeLink-parsed durations equal the `DUR` values in the source file, with tests asserting
      this on the ASC fixtures
- [ ] detected durations equal `t_last - t_first + Δ`
- [ ] nullable-duration semantics (`0` instantaneous, `null` unavailable) implemented and
      documented
- [ ] remaining `offset` consumers reworked to onset/duration arithmetic: `Events.merge`, the
      `compute_event_properties` join and sample selection, `events2segmentation` /
      `segmentation2events`, `measure_events_ratio` (interval-merge retained,
      `sampling_rate` deprecated), the `fill` detector's event mask
- [ ] `minimum_duration` filters on the new duration values, documented as a behavior
      change
- [ ] `offsets=` kept as a permanent input, with `offsets_inclusive` stating the convention
      (`None` default interpreted as inclusive, `DeprecationWarning`, removed v0.34.0)
- [ ] legacy-file loading with pattern detection warning and opt-in correction
- [ ] frame column set and ordering unchanged during the deprecation window (legacy `offset`
      column materialized where derivable), then the single shape flip to the BIDS ordering
      at v0.34.0
- [ ] changelog entry and versioned migration note for the duration value change

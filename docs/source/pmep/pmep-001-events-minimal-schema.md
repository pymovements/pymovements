# PMEP 1: Exact event durations with minimal schema: onset, duration, name

| | |
|---|---|
| **Status** | Draft |
| **Type** | Standards |
| **Author** | Daniel G. Krakowczyk |
| **Created** | 2026-09-16 |
| **Supersedes** | none |

Proposed in issue
[#1677](https://github.com/pymovements/pymovements/issues/1677). Blocks the BIDS events
save/load work ([#1563](https://github.com/pymovements/pymovements/issues/1563)) and closes
[#1083](https://github.com/pymovements/pymovements/issues/1083).

## TL;DR

- The minimal `Events` schema becomes `onset`, `duration`, `name`.
- Every duration pymovements currently produces is one sampling interval short:
  `offset - onset` measures first sample to last. Durations will become `t_last - t_first + Δ`,
  the duration EyeLink reports.
- `duration` gains null semantics: `0` means instantaneous, `null` means unavailable.
- The value change ships as an immediate break in v0.29.0, with a changelog entry and a
  versioned migration note.
- `offset` leaves the minimal schema. Supplied offsets stay stored, their convention declared
  via `offsets_inclusive` or a metadata entry, never guessed. Otherwise `offset` will be an
  on-demand measure with an `inclusive` parameter.
- The constructor will check stored durations against declared offsets. A legacy file,
  undeclared or declared inclusive, will raise, and `durations_from_offsets=True` will derive
  its durations instead.
- `Events.metadata` will record each offset column's convention and the sampling rate. Nothing
  is persisted.

## What it looks like

Constructing the same four events, before and after. Frames built from `durations=` and frames
built from `offsets=` will both have the new shape from v0.29.0 on. Frames built from
`offsets=` or loaded from a file keep the supplied offsets as an additional column, as supplied.

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
# after (v0.29.0): durations are stored exactly, no offset is derived, it is available on demand
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
```

Offsets stay an input alternative, with the convention stated. Supplied offsets are kept as a
column, as supplied. The convention will land in the frame's metadata together with the
sampling rate, so a later save can carry it. Both calls below describe the same four events and
yield the same durations:

```python
# inclusive last-sample offsets: duration = offset - onset + Δ, needs a sampling rate
events = pymovements.Events(
    name=['fixation', 'saccade', 'fixation', 'blink'],
    onsets=[0, 121, 159, 301],
    offsets=[120, 158, 300, 380],
    offsets_inclusive=True,
    sampling_rate=1000,
    trials=[1, 1, 2, 2],
)
events.frame
# the durations change, the supplied offsets stay as an extra column, the convention is recorded
# in metadata. Trial columns first, then the minimal schema, then extras
# ┌───────┬───────┬──────────┬──────────┬────────┐
# │ trial ┆ onset ┆ duration ┆ name     ┆ offset │
# │ 1     ┆ 0     ┆ 121      ┆ fixation ┆ 120    │
# │ 1     ┆ 121   ┆ 38       ┆ saccade  ┆ 158    │
# │ 2     ┆ 159   ┆ 142      ┆ fixation ┆ 300    │
# │ 2     ┆ 301   ┆ 80       ┆ blink    ┆ 380    │
# └───────┴───────┴──────────┴──────────┴────────┘

# exclusive offsets, one past the end: duration = offset - onset, needs no sampling rate
events = pymovements.Events(
    name=['fixation', 'saccade', 'fixation', 'blink'],
    onsets=[0, 121, 159, 301],
    offsets=[121, 159, 301, 381],
    offsets_inclusive=False,
)
events.frame
# the offsets stored as supplied
# ┌───────┬──────────┬──────────┬────────┐
# │ onset ┆ duration ┆ name     ┆ offset │
# │ 0     ┆ 121      ┆ fixation ┆ 121    │
# │ 121   ┆ 38       ┆ saccade  ┆ 159    │
# │ 159   ┆ 142      ┆ fixation ┆ 301    │
# │ 301   ┆ 80       ┆ blink    ┆ 381    │
# └───────┴──────────┴──────────┴────────┘
```

Loading an EyeLink file will give the same values, with durations equal to the file's reported
`DUR` (v0.28 loads them one interval short). `parse_offset=True` will keep the parsed end
timestamps, and the constructor's consistency check will then compare the file's `DUR` against
them:

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

The `Events` constructor gains `durations=`, the offset-convention parameters,
`durations_from_offsets`, `validate` and `metadata`:

```python
Events(
    data: polars.DataFrame | None = None,
    *,
    name: str | list[str] | None = None,
    onsets: list[int | float] | np.ndarray | None = None,
    durations: list[int | float] | np.ndarray | None = None,   # new
    offsets: list[int | float] | np.ndarray | None = None,     # permanent alternative, stays stored
    offsets_inclusive: bool | None = None,   # None means undeclared: raises with an offset column,
                                             # required from v0.29.0
    durations_from_offsets: bool = False,    # replace stored durations by the derivation under the
                                             # declared convention
    sampling_rate: float | None = None,      # written to metadata, must agree with an existing entry
    validate: bool = True,                   # skips the offset/duration consistency check, legacy
                                             # files included, nothing else
    metadata: dict[str, Any] | None = None,  # convention per offset column, sampling rate
    trials: ... = None,
    trial_columns: ... = None,
    time_unit: str | None = None,
)
```

`Events.metadata` will carry the two entries this PMEP names: the convention of each offset
column and the sampling rate. The constructor will write them from `offsets_inclusive=` and
`sampling_rate=`, `from_asc` both for a retained column, the offset measure the convention for
the column it produces, and detectors and loaders the sampling rate. `Events.drop` will remove
a convention entry together with its column. Metadata is excluded from `Events.__eq__`.

The `offset` measure joins the event-measure registry:

```python
offset(*, inclusive: bool = True, sampling_rate: float | None = None) -> polars.Expr
# inclusive=True  (default): onset + duration - sampling_interval, the last-sample timestamp
# inclusive=False:           onset + duration, one past the end, the next onset for adjacent events
```

The `duration` measure stays in the registry and gains the matching parameters:

```python
duration(*, offsets_inclusive: bool = True, sampling_rate: float | None = None) -> polars.Expr
# offsets_inclusive names the convention of the input offset column, as on the constructor
# True (default) is the convention pymovements produces, so offset() then duration() is the identity
# overwrites a stored duration column under the generic collision warning
```

`from_asc` gains a retention parameter for the end timestamps it parses. A future loader that
reports end timestamps will follow the same pattern:

```python
pymovements.gaze.from_asc(file, *, parse_offset: bool = False, ...)
# parse_offset=True:  store the parsed end timestamps as an inclusive offset column, with its
#                     convention entry and the sampling rate, the consistency check then
#                     compares the file's DUR against them
# parse_offset=False: nothing is stored
```

`Dataset.load_event_files` and `Dataset.load` gain the two keywords and forward them to
`Events(frame, ...)` per file. The sampling rate comes from `definition.experiment`. No resource
definition changes:

```python
Dataset.load_event_files(..., offsets_inclusive: bool | None = None, durations_from_offsets: bool = False)
Dataset.load(..., offsets_inclusive: bool | None = None, durations_from_offsets: bool = False)
# offsets_inclusive: None means undeclared, as on the constructor
```

This PMEP defines no on-disk format. The constructor's contract is the metadata dict: a loader
hands it in as `metadata=`, save does the reverse. The BIDS events layout PMEP
([#1563](https://github.com/pymovements/pymovements/issues/1563)) defines the file and the keys.
How `name` and the trial columns map to BIDS files is defined there as well.

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

**Minimal schema.** The minimal `Events` schema becomes `onset`, `duration`, `name` across the
whole codebase. BIDS mandates `onset` first and `duration` second. `name` is the pymovements
event label and has no BIDS column of its own. `offset` leaves the minimal schema and is stored
only where it was supplied explicitly. From v0.29.0 on the frame order will be trial columns,
`onset`, `duration`, `name`, then extras, a supplied `offset` among them. The schema will cover
all discrete-time events, including point events and events of unknown duration. Which event
kinds belong in `Events` rather than `Gaze.messages` is out of scope.

**Duration definition.** Duration becomes the time from the start of the event to its end:

- Events built from samples (detection algorithms, vendor parsers): `t_last - t_first + Δ`, where
  `Δ` is the nominal interval of the sampling rate in effect, equal to `n_samples * Δ` for
  gap-free events. This holds also for events spanning data gaps: data loss inside an event is
  a data-quality measure, not something duration encodes. This matches EyeLink's reported
  `DUR`.
- Events with exact start and end times, not tied to samples (future producers such as
  recording start/stop, calibrations or stimulus presentations): `end - start`, with no `+ Δ`.
  Their timestamps are the event boundaries, not sample positions, so no quantization
  correction applies.

**Nullability.** `duration` gains null semantics, following BIDS: `0` means an instantaneous
point event, `null` means the duration is unavailable. The constructor already accepts `null`
durations since #1637, so what changes is their meaning and how consumers treat them.
Single-sample events will get `Δ`, never `0` (see the quantization note in Rationale).
Duration aggregations will skip `null` rows. Frames with onsets only will be accepted on both
input paths with all-null durations. Missing minimal-schema columns are added as nulls, as
today, and no null `offset` column will appear any more.

**Sample selection.** The half-open interval `[onset, onset + duration)` will select an
event's samples. `onset + duration` equals the next event's onset wherever the next event
starts on the immediately following sample, so adjacent events tile the timeline and share no
boundary sample. A `null` or `0` duration will select no samples. Sample-level operations (AOI
mapping, segmentation, `nullify_event_samples`, the `fill` detector's event mask) move to this
selection and need no sampling interval.

**Producers will compute duration at the source**, where the sampling information lives. The
EyeLink parser will take the reported duration verbatim (the currently discarded
`duration_ms` group). Detection algorithms will compute `t_last - t_first + Δ` from their
`timesteps` and construct `Events(onsets=, durations=)` with no offset column from v0.29.0 on,
writing the sampling-rate entry from the gaze experiment. `from_begaze` is a producer like the
detectors and will build from `durations=`, with `Δ` from the file's rate. Producers get no
`offsets_inclusive` parameter of their own: the convention question only exists for externally
supplied offsets.

**Supplied offsets stay stored.** Offsets supplied via `offsets=` or as an `offset` column in
`data` are kept as an additional column, as supplied. The constructor never removes a column
from `data`. `durations=` and `offsets=` may both be given. An `offset` column needs exactly one
declaration of its convention: `offsets_inclusive=` or a metadata entry. Both given and equal
is fine. Both given and different will raise, neither will raise, and a declaration without an
offset column will raise. `None` on the keyword means undeclared and never selects a
convention. The declaration rule runs unconditionally, `validate` does not skip it. Without a
duration column the durations will be derived under the declared convention. With one, the
stored durations stay and the consistency check judges them. What ends is materializing a null
`offset` column nobody supplied: frames built from `durations=` or by detection algorithms
carry none.

**Metadata.** `Events.metadata` will carry one convention entry per offset column, identified
by the column's name as in a BIDS tabular sidecar, and one sampling-rate entry. The
convention entry is written by `offsets_inclusive=`, by `parse_offset` on `from_asc`
(inclusive) and by the `offset` measure for whatever column it writes, `output_name` included,
rewriting the entry when it overwrites the column. `Events.drop` will remove the entry with the
column, an instance of the column-entry lifecycle the BIDS events layout PMEP (#1563) will
define. Direct `frame` mutation is unsupported for metadata consistency. Several offset columns
with different conventions may coexist. The declaration rule, the consistency check and the
legacy rule apply to the column literally named `offset`, the boundary-changing operations and
the offset measure to any column with an entry. The sampling-rate entry is written by the
constructor's `sampling_rate=`, by detectors from the gaze experiment and by loaders. The rate
resolves in this order: explicit argument, then entry. A constructor value that differs from an
existing entry raises unconditionally. `Gaze.resample` leaves the entry untouched and warns
once when non-empty events carry another rate. Metadata is excluded from `Events.__eq__`,
roundtrip tests compare the two entries explicitly. This PMEP names the two entries and
persists nothing. The BIDS events layout PMEP (#1563) defines the file and the keys.

**Consistency check.** The check will run on public construction with `validate=True` whenever
`offset` and `duration` are both present. Per non-null row, `residual = duration - (offset -
onset)` must be `0` for an exclusive column or `Δ` for an inclusive one, with `Δ` from the
resolved rate rounded to microseconds as the producers do. Without a resolvable rate the check
passes no judgement. Severity follows the rows. If every non-null row contradicts the
declaration, construction raises. If some rows contradict, one warning per construction reports
the row count, the resolved `Δ` with its source, and the residuals seen. Warnings are not
deduplicated. A null duration next to a non-null offset is warned about, never filled. Warnings
may carry a recipe naming `durations_from_offsets=True`.

**Legacy files.** A pre-v0.29.0 file stores `offset` and `duration == offset - onset` on every
row, the residual-0-everywhere case of the check. Without a declaration the declaration rule
raises, declared inclusive the check raises. The message lists, in order:
`durations_from_offsets=True` to derive the durations from the offsets, `offsets_inclusive=False`
to accept the durations as exact, `validate=False` to skip the check, and the Dataset form
`load_event_files(offsets_inclusive=True, durations_from_offsets=True)`. There is no recompute
branch and no `legacy_durations` parameter. The rule is permanent, since the files can always
exist.

**`durations_from_offsets`.** A permanent flag on the constructor, `Dataset.load_event_files`
and `Dataset.load`, default `False`. `True` replaces any stored duration column by the
derivation under the declared convention. It requires an offset column and a declaration, else
it raises. `False` keeps stored durations for the consistency check. The flag is not a
tri-state: the offset-without-duration case derives the durations regardless of the flag.

**The `validate` flag.** Public, default `True`. `validate=False` skips exactly the consistency
check and the legacy rule, nothing else. One constructor, one switch. Rebuild sites pass
`validate=False` and carry the metadata, so a warning is emitted once, at entry: `Events.clone`,
`Events.split`, the per-group filter in `Gaze.detect` and `Events.correct_fixations`.

**Dataset loading.** `Dataset.load_event_files` and `Dataset.load` will forward
`offsets_inclusive` and `durations_from_offsets` to `Events(frame, ...)` per file, with the rate
from `definition.experiment`, and write the sampling-rate entry. No resource definition
changes. Saved event files are derived output.

**Retaining parsed offsets.** `from_asc` alone gains `parse_offset: bool = False`, permanent.
`True` stores the parsed end timestamps as an inclusive `offset` column with its convention
entry and the sampling rate, and the consistency check then compares the file's `DUR` against
them. The parser takes `DUR` verbatim as the duration either way. The offset measure will
reconstruct the same value for sample-built events. In v0.29.0 the guard against a vendor end
timestamp disagreeing with `DUR` is `parse_offset=True` plus the consistency check. A future
loader that reports end timestamps will follow the same pattern. The canonical schema stays
`onset`/`duration`/`name`.

**Persistence.** This PMEP persists nothing. The constructor's contract is the metadata dict: a
loader hands it in as `metadata=`, save does the reverse. The stored duration carries no
convention. A column supplied via `offsets=` is stored as supplied, and its convention lives in
its metadata entry. Until the convention persists, the guard against a vendor end timestamp
disagreeing with `DUR` is `parse_offset=True` plus the consistency check. This PMEP names the
two entries, the BIDS events layout PMEP (#1563) defines the file and the keys.

**The offset measure.** `offset` becomes an on-demand event measure with an `inclusive`
parameter (default `True`, reproducing today's stored offsets). The measure factory never reads
metadata. `inclusive=True` will require a sampling rate, which `Gaze.compute_event_properties`
fills when the caller gives none: the events entry first, the experiment second, the pattern
`Gaze.measure_samples` already uses. `inclusive=False` will need no rate. The orchestrator
writes the convention entry for the column it writes with the value used, `output_name`
included. A collision with an existing column overwrites it under the generic collision warning
of #1690 and rewrites the entry. `inclusive=None` meaning the stored convention is future work.
The measure will compute uniformly over all rows, since the frame does not record whether an
event was built from samples. For events with exact start and end times, `inclusive=False` will
return the end timestamp. `inclusive=True` would subtract `Δ` from a boundary that is not a
sample position, so callers holding such events should use `inclusive=False`.

**The duration measure.** `duration` stays in the registry with the signature from Resulting
signatures. `offsets_inclusive` names the convention of the input offset column, matching the
constructor. The default `True` is the convention pymovements produces, so `offset()` followed
by `duration()` is the identity. A missing offset column raises the ordinary missing-column
error. The measure overwrites stored durations under the generic collision warning. The
constructor flag remains the guarded path and the one named in warnings, the measure is the
documented in-place alternative. `offsets_inclusive=None` reading the entry is future work.

**Frame granularity.** An events frame is produced per recording and carries one sampling
rate. Durations will be computed at the producer, where that rate is known, so they stay exact
however frames are combined later. Combining frames with differing sampling-rate entries will
raise, with a message naming `detect(clear=True)`. Convention entries for the same column that
disagree will raise. A column present on one side only becomes null on the other and keeps its
entry. This binds `Gaze.detect` now: both merge sites carry the detector's metadata into the
gaze's events metadata. No package-level operation for combining events frames exists or is
specified here. A per-event `sampling_rate` column for mixed rates is future work. One rate per
frame also matches BIDS, which ties one events file to one recording.

**Measure plumbing.** `compute_event_properties` already accepts `(name, {kwargs})`. The
internal `EventProcessor` learns to pass the kwargs through to event measures, a non-breaking
internal extension. `Gaze.compute_event_properties` will fill `sampling_rate` for measures that
accept it when the caller gives none, the events entry first, the experiment second. The eager
binding of #1690 is unchanged.

**Remaining offset consumers** move to onset/duration arithmetic:

- `Events.merge_subsequent_close_events`: the gap becomes
  `onset - (previous onset + previous duration)`, the merged duration
  `last onset + last duration - first onset`. The merge and any future boundary-changing
  operation will recompute each offset column via the offset measure in that column's
  convention when its entry exists and a rate resolves where needed. The merge reads the entry
  itself and passes a concrete bool. Otherwise the operation drops the column and its entry,
  once per call, with a warning. This covers offset columns only, by design.
- `compute_event_properties` will join on `['name', 'onset', 'duration']` with
  `nulls_equal=True`, so events with `null` duration keep their rows, and will select samples
  by the half-open interval.
- `events2segmentation` and `segmentation2events` move to the half-open selection.
- `measure_events_ratio` and `events2timeratio` keep merging overlapping intervals (#1713) on
  the new schema and gain a `duration_column` parameter in place of `offset_column`. They stop
  adding one interval to event durations. `sampling_rate` stays, since the total time range
  still spans `t_last - t_first + Δ` over the samples. Events with `null` duration will
  contribute nothing.
- The `fill` detector's event mask moves to the half-open selection, which fixes its
  last-sample leak.

**Duration thresholds.** `minimum_duration` will compare against the new duration values. The
I-VT, I-HMM, fill, microsaccade and blink detectors currently compare `t_last - t_first`
against the threshold and gain the `+ Δ`, so events that previously missed the threshold by
exactly one interval will pass. The blink detector's `maximum_duration` flips the other way:
events that previously passed by exactly one interval will fail. Subtracting one interval from
`minimum_duration` and adding one to `maximum_duration` restores the previous event sets. I-DT
already converts the threshold to a sample count, which is the same convention, so its event
sets will not change.

## Rationale

**Why add one sampling interval (quantization note).** No convention recovers the true
continuous-time duration of an individual event: the true start lies in `(t_first - Δ, t_first]`,
the true end in `[t_last, t_last + Δ)`. Start and end are each known to within one interval, and
duration is their difference, so both errors accumulate: the true duration lies in a window of
width `2Δ`, half the precision of onset and offset, under any convention. The two candidates sit
differently in that window: `offset - onset` is its floor, biased by `-Δ`, while
`t_last - t_first + Δ` is its center, unbiased. For the first fixation of the running example
the true duration lies in `[120, 122)` ms. The new `121` is the center of that window and
EyeLink's reported `DUR`. Today's `120` is its edge, not a more precise value, only a biased one.
Three criteria pick the center independently of bias: additivity (durations sum to recording
length and adjacent events tile), the single-sample event (duration `Δ` instead of a degenerate
`0`), and agreement with the vendor's reported values.

**Why store duration and derive offset only on demand.** Duration is what analyses consume and
what vendors report. Offset depends on a convention (inclusive last sample or one past the end).
Storing the convention-free value and answering the convention question where the offset enters
or leaves, at construction for supplied offsets and in the measure otherwise, removes the
ambiguity that already produced the disagreeing `fill` detector. EyeLink's reported `DUR`
becomes the stored value, so file and frame can no longer contradict each other.

**Why the value change is immediate.** A deprecation window keeps an old and a new API shape
working side by side. A single `duration` column cannot hold both the old and the new number,
so deferring only ships a known-wrong value for five more releases. Bundling it with the
`polars.Duration` change (#1637) in v0.29.0 breaks the events schema once instead of twice and
keeps #1563 from publishing biased durations. The changelog entry and the migration note will
carry the silent numeric shift.

**Why no default convention.** A legacy inclusive file and a correct exclusive file both show a
residual of `0` on every row, and no stored convention will ever exist for the legacy file, so
the constructor refuses to guess. Default inference is dangerous. The governing rule:
pymovements never changes a stored number silently, never keeps a number it knows is wrong, and
never removes a guard while the files it guards can still exist. That rule is why the legacy
raise is permanent.

**Alternatives rejected.**

- *Keep the offset in the minimal schema and document the convention.* Keeps the bias patched
  per consumer rather than fixed at the source, and forces the redundant column into every
  frame and save file, including those built from durations.
- *Store exclusive offsets instead.* Makes `offset - onset` exact, but stores a value no
  vendor reports, breaks verbatim EyeLink round-trips, and leaves the inclusivity question in
  every consumer that compares against sample timestamps.
- *Leave null durations undefined and let the BIDS events work add the semantics.* Breaks the
  same schema twice, and #1563 requires `null` durations for BIDS `n/a` round-trips.
- *Infer the convention from the residual on undeclared columns.* A residual of `0` on every
  row is either a legacy inclusive file or a correct exclusive one, and inference would pick
  silently.
- *Drop and warn when combined frames disagree on a convention.* A dropped column is a silent
  loss of data, failing is the only honest outcome.
- *A tri-state `durations_from_offsets`.* The third value duplicates `False`.
- *Remove the `duration` measure from the registry.* It stays and gains the convention
  parameters, so an in-place recomputation remains available.

## Backwards compatibility

**Immediate breaking change (v0.29.0).** A single break: all detected and parsed durations grow
by one sampling interval, `offset` leaves detected frames and `frame['offset']` is replaced by
the offset measure, the column order changes once to trial columns, `onset`, `duration`,
`name`, extras, `offsets_inclusive` is required with supplied offsets, and `parse_offset`
defaults to `False`.

The blast radius inside the repository: 258 test lines in 19 files, 15 detector call sites,
three `gaze.py` doctest examples and zero tutorials or notebooks. The doctests are the
build-breaking item. Every call that supplies offsets declares their convention:

```python
pm.Events(name='blink', onsets=[2], offsets=[3])                                              # v0.28
pm.Events(name='blink', onsets=[2], offsets=[3], offsets_inclusive=True, sampling_rate=1000)  # v0.29.0
pm.Events(name='blink', onsets=[2], durations=[2])                                            # or durations
```

**Release gate.** v0.29.0 does not ship until the sequence `Gaze.detect`, `Dataset.save_events`,
`Dataset.load_event_files` passes with no metadata file and no extra keyword.

**Offsets stay as input, with an explicit convention.** `offsets=` remains a permanent
alternative to `durations=`. The supplied offsets stay stored as an `offset` column. Without a
duration column the duration is derived at construction under the declared convention. With
one, the stored durations stay and the consistency check judges them, and
`durations_from_offsets=True` replaces them by the derivation. `offsets_inclusive` states the
convention of the supplied offsets, a metadata entry can state it instead:

- `True`: inclusive last-sample timestamps, `duration = offset - onset + Δ`. Requires a
  sampling rate, resolved from the explicit argument, then the metadata entry.
- `False`: one past the end, `duration = offset - onset`. Needs no rate.

**Deprecated v0.29.0, removed v0.34.0.** `offset_column` on `Gaze.measure_events_ratio` and
`events2timeratio` refers to a column that leaves the schema. `duration_column` replaces it.

**Migration note (versioned).** Documents the value change, the `minimum_duration` and
`maximum_duration` threshold adjustments, the legacy recipe in constructor and Dataset form,
the replacement of `frame['offset']` by the offset measure, and the `parse_offset` default.

## Implementation

- [ ] `onset`/`duration`/`name` schema, `durations=` path, `offsets_inclusive` required,
      supplied offsets stay stored, `data` accepted without `offset`, frame order flipped once
- [ ] `metadata` entries: per-column convention and `sampling_rate`, the declaration and
      sampling-rate raises unconditional, `Events.drop` removes the entry
- [ ] `validate` flag, four rebuild sites pass metadata and `validate=False`, parametrised
      equality test over the four
- [ ] `durations_from_offsets` on the constructor and both Dataset loaders, `offsets_inclusive`
      on both Dataset loaders
- [ ] consistency check with the severity rule and recipes, legacy-file message
- [ ] `offset` measure with `inclusive` and `sampling_rate`, entry written on output
- [ ] `duration` measure with `offsets_inclusive` and `sampling_rate`
- [ ] `compute_event_properties` sampling-rate injection (entry, then experiment)
- [ ] `Gaze.detect`: metadata merge at both sites, raises on disagreeing entries, `resample`
      warning
- [ ] `EventProcessor` passes measure kwargs through (#1690)
- [ ] EyeLink durations equal the file's DUR, tested on the ASC fixtures, `parse_offset` on
      `from_asc`, default False
- [ ] BeGaze events built from `durations=`
- [ ] detected durations equal `t_last - t_first + Δ`
- [ ] `minimum_duration` compares against the new durations in I-VT, I-HMM, fill,
      microsaccades and blink, `maximum_duration` in blink
- [ ] null duration semantics
- [ ] offset consumers reworked: `merge_subsequent_close_events` recomputes or drops offset
      columns with their entries, `compute_event_properties` join,
      `events2segmentation`/`segmentation2events`,
      `measure_events_ratio`/`events2timeratio` with `duration_column`, `fill`
- [ ] changelog entry and versioned migration note (value change, thresholds, legacy recipe in
      constructor and Dataset form, `frame['offset']` replacement, `parse_offset` default)

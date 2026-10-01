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
events = Events(
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
events = Events(
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

Offsets stay an input alternative, with the convention stated. The convention lands in the
frame's metadata together with the sampling rate. Both calls below describe the same four
events and yield the same durations:

```python
# inclusive last-sample offsets: duration = offset - onset + Δ, needs a sampling rate
events = Events(
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
events = Events(
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
`durations_from_offsets` and `validate`:

```python
Events(
    data: polars.DataFrame | None = None,
    *,
    name: str | list[str] | None = None,
    onsets: list[int | float] | np.ndarray | None = None,
    durations: list[int | float] | np.ndarray | None = None,   # new
    offsets: list[int | float] | np.ndarray | None = None,     # permanent alternative, stays stored
    offsets_inclusive: bool | None = None,   # None means undeclared: raises with an offset column
                                             # that has no metadata entry
    durations_from_offsets: bool = False,    # replace stored durations by the derivation under the
                                             # declared convention
    sampling_rate: float | None = None,      # written to metadata, must agree with an existing entry
    validate: bool = True,                   # skips the offset/duration consistency check, legacy
                                             # files included, nothing else
    metadata: dict[str, Any] | None = None,  # gains convention per offset column, sampling rate
    trials: ... = None,
    trial_columns: ... = None,
    time_unit: str | None = None,
)
```

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

Detection algorithms and `segmentation2events` gain a required `sampling_rate` keyword.
`Gaze.detect` fills it from the experiment.

This PMEP defines no on-disk format. How `name` and the trial columns map to BIDS files is
defined in the BIDS events layout PMEP
([#1563](https://github.com/pymovements/pymovements/issues/1563)).

## Motivation

`Events.duration` is currently derived as `offset - onset`. All producers store the offset as the
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
- Saved events files publish the short durations, and every downstream analysis that reads them
  inherits the bias.

## Specification

**Minimal schema.** The minimal `Events` schema becomes `onset`, `duration`, `name` across the
whole codebase. BIDS mandates `onset` first and `duration` second. `name` is the pymovements
event label and has no BIDS column of its own. `offset` leaves the minimal schema and is stored
only where it was supplied explicitly. Trial columns stay first. Within the minimal schema the
order changes from `name`, `onset`, `offset` to `onset`, `duration`, `name`, then extras follow,
a supplied `offset` among them. The schema will cover events built from samples, including
point events and events of unknown duration. Events with exact boundaries not tied to samples
is future work.

**Duration definition.** Duration becomes the time from the start of the event to its end,
`t_last - t_first + Δ`, where `Δ` is the nominal interval of the sampling rate in effect. This
equals `n_samples * Δ` for gap-free events and holds also for events spanning data gaps: data
loss inside an event is a data-quality measure, not something duration encodes. This matches
EyeLink's reported `DUR`.

**Nullability.** `duration` gains null semantics, following BIDS: `0` means an instantaneous
point event, `null` means the duration is unavailable. The constructor already accepts `null`
durations, so what changes is their meaning and how consumers treat them.
Single-sample events will get `Δ`, never `0` (see the quantization note in Rationale).
Duration aggregations will skip `null` rows. Frames with onsets only will be accepted on both
input paths with all-null durations. Missing minimal-schema columns are added as nulls, as
today.

**Sample selection.** The half-open interval `[onset, onset + duration)` will select the samples
of an event. `onset + duration` equals the next event's onset wherever the next event
starts on the immediately following sample, so adjacent events tile the timeline and share no
boundary sample. A `null` or `0` duration will select no samples. Sample-level operations (AOI
mapping, segmentation, `nullify_event_samples`, the `fill` detector's event mask, the time
series plot's event shading) move to this selection and need no sampling interval. The
selection assumes timestamps on the nominal sampling grid: a sample arriving earlier than
`t_last + Δ` would fall into the preceding event. A tolerance for off-grid timestamps is future
work.

**Producers will compute duration at the source**, where the sampling information lives. The
EyeLink parser will take the reported duration verbatim (the currently discarded
`duration_ms` group). Detection algorithms gain a required `sampling_rate` keyword, the source
of `Δ` and of the sampling-rate entry they write. They will compute `t_last - t_first + Δ` from
their `timesteps` and construct `Events(onsets=, durations=)` with no offset column from
v0.29.0 on. `Gaze.detect` will fill `sampling_rate` from the experiment when the caller gives
none and will raise on a gaze without one. Direct calls pass `sampling_rate` explicitly.
Estimating the sampling rate from `timesteps` is future work. `from_begaze` is a producer like
the detectors and will build from `durations=`, with `Δ` from the file's sampling rate.
Producers get no `offsets_inclusive` parameter of their own: the convention question only
exists for externally supplied offsets.

**Supplied offsets stay stored.** Offsets supplied via `offsets=` or as an `offset` column in
`data` will be kept as an additional column, as supplied. The constructor will never remove a
column from `data`. `durations=` and `offsets=` may both be given. What ends is materializing
a null `offset` column if not supplied: frames built from `durations=` or by detection
algorithms will carry none. What happens to durations next to a stored offset column follows
the construction rules below.

**Metadata.** This PMEP adds two entries to `Events.metadata`: one convention entry per offset
column, identified by the column's name as in a BIDS tabular sidecar, and one sampling-rate
entry. The convention entry will be written by `offsets_inclusive=`, by `parse_offset` on
`from_asc` and by the `offset` measure for the column it writes. Several offset columns with
different conventions may coexist. The declaration rule, the consistency check and the legacy
rule apply to the column literally named `offset`, the boundary-changing operations and the
offset measure to any column with an entry. The sampling-rate entry will be written by the
constructor's `sampling_rate=`, by detectors, by loaders and by `Gaze.resample`, which sets it
to the new sampling rate. The entry names the sampling grid the frame currently lives on, not
the grid its durations were built on. Combining frames whose entries disagree, as `Gaze.detect`
does with the existing events, will raise. This PMEP persists nothing: a loader hands the dict
in as `metadata=`, save does the reverse, and the BIDS events layout PMEP defines the file and
the keys.

**Construction rules.** An `offset` column needs its convention declared, by
`offsets_inclusive=` or by a metadata entry for the column, both when they agree. `None` on
the keyword means undeclared and never selects a convention. The sampling rate resolves from
`sampling_rate=`, then from the metadata entry. The preconditions run unconditionally on
public construction, `validate` never skips them, and each one raises:

| condition | outcome |
|---|---|
| offset convention declared, no `offset` column | raise |
| `offsets_inclusive=` and the metadata entry both given and different | raise |
| `offset` column, convention undeclared | raise, the message lists the legacy recipes |
| `sampling_rate=` differs from an existing entry | raise |
| `durations_from_offsets=True` without `offset` column | raise |

Once the preconditions pass, the outcome depends on which columns are present and on the two
flags. A derivation under an inclusive declaration needs the sampling rate and raises without
one:

| `offset` column | `duration` column | `durations_from_offsets` | `validate` | outcome |
|---|---|---|---|---|
| no | no | `False` | any | `duration` column of nulls, no `offset` column |
| no | yes | `False` | any | keep the durations |
| yes | no | any | any | derive the durations under the declared convention |
| yes | yes | `True` | any | replace the durations by the derivation |
| yes | yes | `False` | `False` | keep the durations, no check |
| yes | yes | `False` | `True` | keep the durations, run the consistency check |

**Consistency check.** Per non-null row, `residual = duration - (offset - onset)`. An exclusive
column requires `0`. An inclusive column requires `Δ` when a sampling rate resolves. Without
one the residuals of an inclusive column must all be equal and greater than zero, and the
constant is never written to the sampling-rate entry. The check therefore needs no sampling
rate. Severity follows the non-null rows:

| rows contradicting the declaration | outcome |
|---|---|
| none | pass |
| all | raise, the message lists the legacy recipes |
| some | one warning per construction |

A null `duration` beside a non-null `offset` counts for neither row. It is warned about once
and never filled.

**Legacy files.** A pre-v0.29.0 file stores `offset` and `duration == offset - onset` on every
row, the residual-0-everywhere case. Read off the tables:

| call | rule | outcome |
|---|---|---|
| no keywords | convention undeclared | raise |
| `offsets_inclusive=True` | check, all rows contradict | raise |
| `offsets_inclusive=True, durations_from_offsets=True` | replace | durations grow by `Δ` |
| `offsets_inclusive=False` | check, no row contradicts | durations kept as exact |
| `offsets_inclusive=True, validate=False` | keep, no check | short durations kept, opted out |

The replace row needs a sampling rate, which the Dataset form takes from the experiment. The
raise messages list the recipes. The rule is permanent, since the files can always exist.

**`durations_from_offsets`.** A permanent flag on the constructor, `Dataset.load_event_files`
and `Dataset.load`, default `False`, with the behaviour of the outcome table.

**Retaining parsed offsets.** The EyeLink parser takes `DUR` verbatim as the duration whether
or not `parse_offset` is set. In v0.29.0 `parse_offset=True` plus the consistency check is the
guard against a vendor end timestamp disagreeing with `DUR`.

**The offset measure.** `offset` becomes an on-demand event measure with an `inclusive`
parameter (default `True`, reproducing today's stored offsets). `inclusive=True` will require
a sampling rate, which `Gaze.compute_event_properties` will fill when the caller gives none:
the events entry first, the experiment second. `inclusive=False` will need no sampling rate.
`inclusive=None` meaning the stored convention is future work.

**Remaining offset consumers** move to onset/duration arithmetic:

- `Events.merge_subsequent_close_events`: the gap of adjacent events becomes `0` instead of
  `Δ`, so a `max_gap` that relied on adjacency showing as one interval shrinks by `Δ`. The
  merge, `Gaze.resample` and any future boundary-changing operation will recompute each offset
  column via the offset measure in that column's convention when its entry exists and a
  sampling rate resolves where needed. Otherwise the operation will drop the column and its
  entry with a warning. This covers offset columns only, by design.
- `compute_event_properties` keeps the rows of events with `null` duration and selects samples
  by the half-open interval.
- `events2segmentation` gains the same `duration` parameter in place of `offset_column`.
  `segmentation2events` is a producer: it builds events from sample runs and will compute
  `t_last - t_first + Δ` like the detectors, so it gains a required `sampling_rate` keyword too.
- `measure_events_ratio` and `events2timeratio` keep merging overlapping intervals on the new
  schema and gain a `duration: str | pl.Expr` parameter in place of `offset_column`. They stop
  adding one interval to event durations. `sampling_rate` stays, since the total time range
  still spans `t_last - t_first + Δ` over the samples. Events with `null` duration will
  contribute nothing.

**Duration thresholds.** `minimum_duration` will compare against the new duration values. The
I-VT, I-HMM, fill, microsaccade and blink detectors currently compare `t_last - t_first`
against the threshold and gain the `+ Δ`, so events that previously missed the threshold by
exactly one interval will pass. The blink detector's `maximum_duration` flips the other way:
events that previously passed by exactly one interval will fail. Adding one interval to both
`minimum_duration` and `maximum_duration` restores the previous event sets. I-DT
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
length and adjacent events tile), the single-sample event (duration `Δ` instead of an ambiguous
`0`), and agreement with the vendor's reported values.

**Why store duration and derive offset only on demand.** Duration is what analyses consume and
what vendors report. Offset depends on a convention (inclusive last sample or one past the end).
Storing the convention-free value and answering the convention question where the offset enters
or leaves, removes the ambiguity that already produced the disagreeing `fill` detector. EyeLink's
reported `DUR` becomes the stored value, so file and frame can no longer contradict each other.

**Why the value change is immediate.** A deprecation window keeps an old and a new API shape
working side by side. A single `duration` column cannot hold both the old and the new number,
so deferring only ships a known-wrong value for five more releases. Bundling it with the
`polars.Duration` change in v0.29.0 breaks the events schema once instead of twice and keeps
#1563 from publishing biased durations. The changelog entry and the migration note will carry
the silent numeric shift.

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

## Backwards compatibility

**Immediate breaking change (v0.29.0).** A single break: all detected and parsed durations grow
by one sampling interval, `offset` leaves detected frames and `frame['offset']` is replaced by
the offset measure, the minimal-schema order changes once to `onset`, `duration`, `name` behind
the trial columns, a convention declaration is required with supplied offsets, `parse_offset`
defaults to `False`, direct detector calls need `sampling_rate=`, and `Gaze.detect` raises on a
gaze without an experiment sampling rate.

Every call that supplies offsets declares their convention:

```python
Events(name='blink', onsets=[2], offsets=[3])                                                 # v0.28
Events(name='blink', onsets=[2], offsets=[3], offsets_inclusive=True, sampling_rate=1000)     # v0.29.0
Events(name='blink', onsets=[2], durations=[2])                                               # or durations
```

**Release gate.** v0.29.0 does not ship until the sequence `Dataset.detect`,
`Dataset.save_events`, `Dataset.load_event_files` passes with no metadata file and no extra
keyword.

**Own output with an offset column.** Nothing is persisted, so a saved frame that carries an
`offset` column, from `parse_offset=True`, from the offset measure or from `offsets=`, reloads
only with `offsets_inclusive` given: `True` for the first two, the declared value for the
third. The consistency check then passes. The BIDS events layout PMEP (#1563) closes this gap
with the sidecar. Until then the migration note names it next to the `frame['offset']`
replacement.

**Offsets stay as input, with an explicit convention.** `offsets=` remains a permanent
alternative to `durations=`, with the convention declared as in Specification.

**Deprecated v0.29.0, removed v0.34.0.** `offset_column` on `Gaze.measure_events_ratio`,
`events2timeratio` and `events2segmentation` refers to a column that leaves the schema.
`duration: str | pl.Expr` replaces it.

**Migration note (versioned).** Documents the value change, with the parsed EyeLink durations
named separately from the detected ones since the same ASC file yields durations one `Δ`
longer with no error, the `minimum_duration`, `maximum_duration` and merge gap threshold
adjustments, the legacy recipe in constructor and Dataset form, the replacement of
`frame['offset']` by the offset measure, the reload keyword for own output with an offset
column, and the `parse_offset` default.

## Implementation

One issue per line, drafted once the PMEP is accepted:

- [ ] `Events` constructor: schema and order, `durations=`, the construction rules, the
      consistency check, `durations_from_offsets`, `validate`, the legacy message
- [ ] `Events.metadata` entries: per-column convention and sampling rate
- [ ] offset and duration measures, `compute_event_properties` sampling-rate injection
- [ ] `Gaze.detect` and `Gaze.resample`: detector `sampling_rate`, metadata merge, entry update,
      offset columns on resample
- [ ] Dataset loaders: the two keywords, `sampling_rate` from the experiment
- [ ] EyeLink and BeGaze parsers: `DUR` verbatim, `parse_offset`, `durations=`
- [ ] offset consumers, duration thresholds, null duration semantics
- [ ] changelog entry and versioned migration note per Backwards compatibility

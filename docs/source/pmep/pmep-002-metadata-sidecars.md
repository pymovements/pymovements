# PMEP 2: Metadata sidecars for saved files

| | |
|---|---|
| **Status** | Draft |
| **Type** | Standards |
| **Author** | Daniel G. Krakowczyk |
| **Created** | 2026-10-01 |
| **Supersedes** | none |

Requires [PMEP 1](https://github.com/pymovements/pymovements/pull/1730), which adds two entries
to `Events.metadata` and persists nothing. The BIDS events layout
([#1563](https://github.com/pymovements/pymovements/issues/1563)) will extend the methods
defined here.

## TL;DR

- Saved `Events`, `Participants` and `Phenotype` files will share one metadata sidecar:
  `<stem>.json` next to the data file, in the BIDS tabular shape, always written.
- `Events` gains `save` and `load` for tsv, csv and feather. The two metadata entries of PMEP 1
  get their keys, `SamplingFrequency` and per offset column `OffsetsInclusive`.
- The dict is the file. Load puts the sidecar into `metadata` as it is, save writes it back as
  it is. Where an entry contradicts the frame, the frame wins at save, with a warning.
- A `pymovements` object in the sidecar carries the schema version, initially `0.1.0`.
- tsv takes the BIDS defaults: tab, seconds, `n/a`. csv needs an explicit time unit. feather
  stays native.
- `verify_bids` reports nonconformities, as in `Phenotype`.
- Ships in v0.30.0. `Participants` and `Phenotype` keep their released behavior.

## What it looks like

Saving and reloading events on a dataset, before and after:

```python
# before (v0.29.0): only the frame is written
dataset.save_events(extension='csv')    # milliseconds, with no record of the unit
dataset.load_event_files(extension='csv', offsets_inclusive=True)
# a stored offset column needs its convention restated by hand, and trial_columns is None on
# the loaded events
```

```python
# after (v0.30.0): the metadata travels with the file
dataset.save_events(extension='tsv')
dataset.load_event_files(extension='tsv')
```

`Events` gains the two methods the dataset calls per file. The events are the running example
of PMEP 1 with a source file added:

```python
events = Events(
    name=['fixation', 'saccade', 'fixation', 'blink'],
    onsets=[0, 121, 159, 301],
    offsets=[120, 158, 300, 380],
    offsets_inclusive=True,
    sampling_rate=1000,
    trials=[1, 1, 2, 2],
    metadata={'sources': ['raw/subject.asc']},
)
events.save('events.tsv')             # writes events.tsv and events.json
events = Events.load('events.tsv')    # reads both, no keyword needed
```

`events.tsv` holds the time columns as seconds (tab-separated, shown aligned):

```text
trial  onset  duration  name      offset
1      0.0    0.121     fixation  0.12
1      0.121  0.038     saccade   0.158
2      0.159  0.142     fixation  0.3
2      0.301  0.08      blink     0.38
```

`events.json` describes each column at the top level and the file beside them:

```json
{
    "trial": {"Format": "integer"},
    "onset": {"Format": "number", "Units": "s"},
    "duration": {"Format": "number", "Units": "s"},
    "name": {"Format": "string"},
    "offset": {"Format": "number", "Units": "s", "OffsetsInclusive": true},
    "SamplingFrequency": 1000,
    "sources": ["raw/subject.asc"],
    "trial_columns": ["trial"],
    "pymovements": {
        "schema": "events",
        "schema_version": "0.1.0",
        "version": "0.30.0",
        "file": "events.tsv"
    }
}
```

**TODO (Daniel):** the decision record does not name the key that holds the data file name
inside the `pymovements` object. `file` is a placeholder.

Saving the same events as `events.feather` instead writes no `Format` and no `Units`, because
feather stores the dtypes itself:

```json
{
    "offset": {"OffsetsInclusive": true},
    "SamplingFrequency": 1000,
    "sources": ["raw/subject.asc"],
    "trial_columns": ["trial"],
    "pymovements": {
        "schema": "events",
        "schema_version": "0.1.0",
        "version": "0.30.0",
        "file": "events.feather"
    }
}
```

After `Events.load('events.tsv')` the dict is the sidecar:

```python
events.metadata
# {
#     'trial': {'Format': 'integer'},
#     'onset': {'Format': 'number', 'Units': 's'},
#     'duration': {'Format': 'number', 'Units': 's'},
#     'name': {'Format': 'string'},
#     'offset': {'Format': 'number', 'Units': 's', 'OffsetsInclusive': True},
#     'SamplingFrequency': 1000,
#     'sources': ['raw/subject.asc'],
#     'trial_columns': ['trial'],
#     'pymovements': {'schema': 'events', 'schema_version': '0.1.0', 'version': '0.30.0',
#                     'file': 'events.tsv'},
# }
events.trial_columns           # ['trial'], a view on metadata['trial_columns']
events.frame.schema['onset']   # Duration(time_unit='us'), built from Format and Units
```

**TODO (Daniel):** which value does `sources` hold after load? "The dict is the file" gives the
sidecar's value, as shown. The `add_source` docstring in
[#1655](https://github.com/pymovements/pymovements/pull/1655) says a loaded sidecar's `sources`
is not forwarded and the file that was read is recorded.

## Resulting signatures

`Events` gains `save` and `load`:

```python
Events.save(
    path: str | Path,                          # the extension selects tsv, csv or feather
    *,
    time_unit: str | None = None,              # unit of all time columns in a text file,
                                               # for this call only
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = 'REQUIRED',
    metadata_path: str | Path | None = None,   # None means <stem>.json next to the data file
) -> None

Events.load(
    path: str | Path,
    metadata: str | Path | dict[str, Any] | None = None,   # a path or dict replaces <stem>.json
    *,
    trial_columns: list[str] | str | None = None,   # these four override their sidecar entry,
    time_unit: str | None = None,                   # with a warning that names both values
    offsets_inclusive: bool | None = None,
    sampling_rate: float | None = None,
    durations_from_offsets: bool = False,           # forwarded to the constructor, see PMEP 1
) -> Events
```

**TODO (Daniel):** the decision record gives `Events.load` a `metadata` parameter and "explicit
keywords" without listing them. The four shown are the constructor keywords with a sidecar
entry, and `durations_from_offsets` is what PMEP 1 gives the Dataset loaders. Open: `validate`,
a `verify_bids` on load as `Phenotype.load` has it, and the pass-through parameters of
`Phenotype` (`separator`, `read_csv_kwargs`, `write_csv_kwargs`, `metadata_encoding`).

The existing writers and the loader delegate to the two methods:

```python
Dataset.save_events(..., extension: str = 'feather', time_unit: str | None = None,
                    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = 'REQUIRED')
Dataset.save(..., extension: str = 'feather', time_unit: str | None = None,
             verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = 'REQUIRED')
# extension gains 'tsv', the two keywords are new and go to Events.save per file

Dataset.load_event_files(events_dirname: str | None = None, extension: str = 'feather', ...)
# no new keyword beyond PMEP 1, delegates to Events.load per file

Gaze.save_events(path: Path, *, verbose: int = 1) -> None
# signature unchanged, delegates to Events.save, so tsv is accepted and the sidecar is written
```

The Dataset methods do not expose the sidecar path.

**TODO (Daniel):** `Dataset.save` passes one `extension` to `save_events` and
`save_preprocessed`, and `save_preprocessed` accepts only `feather` and `csv`. The decision
record gives `Dataset.save` tsv without saying what happens to the samples. It is also silent on
whether `Gaze.save_events` gains `time_unit` and `verify_bids`.

`Participants` and `Phenotype` keep their signatures. Their sidecars gain the `pymovements`
object, and both classes follow the rules for a contradicting `Format` and for an entry without
a column.

**File structure.** Schema `events`, initial schema version `0.1.0`, carried in the sidecar:

```text
<stem>.tsv | <stem>.csv | <stem>.feather    the data file
<stem>.json                                 the sidecar, a JSON object
```

| key | level | value |
|---|---|---|
| `<column name>` | top | object that describes the column |
| `Format` | column | BIDS format: `string`, `number`, `integer`, `bool`, `index`, `label` |
| `Units` | column | unit of the column, on a time column the unit it is written in |
| `OffsetsInclusive` | column | boolean, marks an offset column and states its convention |
| `SamplingFrequency` | top | sampling rate in Hz |
| `sources` | top | list of source files |
| `trial_columns` | top | list of trial column names |
| `pymovements` | top | the stamp: `schema`, `schema_version`, `version`, data file name |

Any other key is kept as it is, on load and on save.

## Motivation

PMEP 1 adds an offset convention and a sampling rate to `Events.metadata` and persists nothing.
A saved frame with an `offset` column therefore reloads only when the caller restates the
convention, and PMEP 1 names a later PMEP on metadata sidecars as the one that closes this gap.

The gap is wider than these two entries:

- `Events` has no `save` and no `load`. Two writers exist, `Gaze.save_events` and
  `Dataset.save_events`, and both accept only `feather` and `csv`.
- The csv writers convert every Duration column to milliseconds and record the unit nowhere.
- `Dataset.load_event_files` constructs `Events(frame)` without `trial_columns`, so a saved
  and reloaded dataset loses them.
- `Dataset.load_event_files` accepts `tsv` and reads it with the comma default of
  `polars.read_csv`. No method writes tsv, so a tsv events file has never round-tripped.
- `Participants` and `Phenotype` already write a BIDS sidecar from their `metadata` dict.
  Consistency across classes is preferred over a second format for events.

## Specification

The mechanism is generic. It is applied to `Events`, `Participants` and `Phenotype`, and the
rules below hold for all three unless they name a class.

**The sidecar file.** Every save will write `<stem>.json` next to the data file. There is no
switch to turn it off. `metadata_path` on save gives a custom path, and `metadata` on load takes
a path or a dict, as in `Phenotype` today. `Participants.save` keeps its released default
`participants.json`. The sidecar has the BIDS tabular shape: one object per column at the top
level, keyed by the column name, and the file-level keys beside them. There is no wrapper object
around the columns.

**The stamp.** The top-level `pymovements` object holds only the stamp: the schema name, the
schema version, the package version and the name of the data file. It is kept in the dict after
load for inspection and overwritten at every save. Two data files with the same stem share one
sidecar path, so load will warn when the stamp names a different data file than the one being
read.

**The schema version** has three parts. Below `1.0.0` the minor is the breaking position and the
patch the additive one, as in the package. Schema `1.0.0` will be declared with pymovements
`1.0.0`. Load compares the stamp with the version it implements:

| stamp found | load |
|---|---|
| newer minor | refuses, the message names the version found |
| same minor, newer patch | warns, proceeds and keeps the keys it does not know |
| older | always reads |
| no stamp | version 0, loaded as descriptive entries with one warning |

**TODO (Daniel):** what does "loaded as descriptive entries" mean for the entries of an
unstamped sidecar, and does the rule hold for `Participants` and `Phenotype`? Their released
sidecars and every third-party BIDS sidecar carry no stamp, and their `Format` entries drive the
cast today.

**The dict.** Four rules connect `metadata`, the sidecar and the frame:

1. The dict is the file. Load puts the sidecar into the dict as it is, and save writes the dict
   as it is.
2. For text files the loader builds the dtypes from `Format` and `Units`, and the writer fills
   both in where they are missing. A Duration column is written as `number`. Feather needs
   neither. The writer never adds `Units` to a feather sidecar, and an entry that is already in
   the dict is carried.
3. On contradiction the frame wins. A `Format` that does not fit the dtype of its column is
   replaced at save, with a warning.
4. `events.trial_columns` is a view on `metadata['trial_columns']`. The constructor keyword
   writes the entry. A keyword that differs from an entry in `metadata=` raises, as PMEP 1 has
   it for its two entries.

Assigning a whole new dict to `metadata` also drops the trial columns, and nothing raises
afterwards.

**TODO (Daniel):** does save write what it fills in, replaces and stamps back into the dict in
memory, or only into the file? The decision record says the stamp is "overwritten at every
save" and that `time_unit=` holds for "that call only".

**Units.** `Units` on a time column states the unit the column is written in as a number. A
column whose `Units` is a time unit loads as a Duration. On save to a text file the unit of a
time column resolves in this order: `time_unit=`, which applies to all time columns and to that
call only, then the `Units` entry of the column, then the default of the format. Columns with
different entries are therefore written in different units.

**Boundaries.**

| situation | outcome |
|---|---|
| keyword on `load` differs from its sidecar entry | keyword wins, warning names both values |
| `metadata=` given on `load` | replaces the sidecar |
| column entry without its column | kept, warning at construction, load and save |
| top-level key equals a column name, value is not an object | save raises, load warns |
| sidecar and dataset definition both give a sampling rate | sidecar wins, no warning |
| direct change to the dict or the frame | not checked until the next save |

The message of the raise names the fix. The definition's sampling rate is used only when the
sidecar has none. The `trial_columns` setter validates its value as the constructor does, and
save applies the same check.

**TODO (Daniel):** how is a column entry without its column told apart from a file-level key
whose value is an object, such as `MeasurementToolMetadata` in a `Phenotype` sidecar or a free
user key?

**Formats.** The extension of the path selects the format:

| | separator | time unit | verification |
|---|---|---|---|
| tsv | tab | seconds by default | findings and not-implemented notices |
| csv | comma | must be specified | findings for separator and unit |
| feather | none | native Duration | none |

tsv is new for saving and takes the BIDS defaults: tab, seconds and `n/a` for nulls. csv has no
default unit. The unit comes from `time_unit=` or from `Units` entries, and `Events.save` raises
without one. The message names `time_unit='ms'` as the value that reproduces today's files. A
csv file without a sidecar is read as milliseconds.

**TODO (Daniel):** which unit does load assume for a tsv file without a sidecar? The decision
record covers csv only. It is also silent on `extension='txt'`, which `Dataset.load_event_files`
accepts today.

**Nested columns** raise for text files, and the message names `events.unnest()`. Feather stores
them natively. The text format is thereby specified on flat columns only, independent of how
nested columns are stored.

**Verification.** `verify_bids` works as in `Phenotype`: `'REQUIRED'`, the default on save,
warns for each finding, `True` raises and `False` is silent. For events this PMEP defines five
checks, all of them BIDS requirements:

- `onset` and `duration` are present
- numeric time columns are in seconds
- nulls are written as `n/a`
- the separator is a tab
- a top-level key that matches a column name is an object

Three more requirements cannot be met before the BIDS layout PMEP: the onset reference, the file
naming and the mapping of `name`. They are reported as not-implemented notices, which warn and
never raise.

**Events.** `Events.save` and `Events.load` are new, with the schema name `events`. The sidecar
gives PMEP 1's two entries their keys: the sampling-rate entry is `SamplingFrequency` at the top
level, and the convention entry is the boolean `OffsetsInclusive` in the object of its offset
column. `sources` is written at the top level as
[#1655](https://github.com/pymovements/pymovements/pull/1655) defines it, and `trial_columns`
beside it. Only `Events.drop` removes a column's entry. `Events.unnest` moves it to the component
columns, and methods that add or rename columns maintain the entry of the column.

**Participants and Phenotype** already model the BIDS sidecar in `metadata`, and their
signatures and released behavior stay. Three things are aligned: their sidecars gain the stamp,
an entry without a column warns and is kept, and a `Format` that contradicts the frame is
replaced at save with a warning.

**TODO (Daniel):** the decision record gives the stamp to `Participants` and `Phenotype`
without naming their `schema` values and initial schema versions.

## Rationale

**Why the BIDS tabular shape.** BIDS conformity is the target of the design, and defaults should
conform. A user may deviate and gets a warning or a raise through verification. The BIDS common
principles (v1.11.1) let a data dictionary hold column fields "in addition to any other metadata
one wishes to include that describe the file as a whole", and they require: "If a field name
included in the data dictionary matches a column name in the TSV file, then that field MUST
contain a description of the corresponding column". The first sentence allows the file-level
keys and the `pymovements` object. The second is the reason save raises on a key that equals a
column name without being an object. A probe with bids-validator-deno 3.0.2 and 2.4.1, on a raw
and a derivative dataset with identical results, confirmed the shape:

| sidecar content | validator |
|---|---|
| top-level `pymovements` object | clean |
| top-level lowercase `sources`, free user key | clean |
| extra field inside a column object | clean |
| `SamplingFrequency` in an events sidecar | clean |
| description for a nonexistent column | clean |
| `columns` wrapper around the column objects | warning `TSV_ADDITIONAL_COLUMNS_UNDEFINED` |
| `Units: "ms"` on `onset` of a BIDS events file | warning `TSV_COLUMN_TYPE_REDEFINED` |

The validator inspects only the keys it knows, so clean means not looked at.

**Why the dict is the file.** Every variant in which load removed derived fields from the dict
lost information and needed an exception to get it back: first `label`, which shares its dtype
with `string`, then the unit of a file that is loaded and saved again. Nothing is removed, so
nothing needs restoring.

**Why entries without a column are kept.** pymovements never removes a metadata entry on its
own. PMEP 1 already keeps a convention entry without its column, and BIDS treats a description
for a nonexistent column as other metadata.

**Why csv has no default unit.** Today's csv files hold milliseconds. A seconds default would
change every number by a factor of 1000 for external readers of these files, without an error.
tsv has no such history, and BIDS requires seconds there.

**Why the onset reference is only a notice.** BIDS `onset` counts seconds from the start of the
recording, and pymovements onsets are tracker timestamps. The mapping needs a reference
timestamp, which belongs to the BIDS layout.

**Alternatives rejected.**

- *A `columns` wrapper for the column objects.* Not the BIDS shape, and the validator no longer
  finds the descriptions.
- *The offset convention, `sources` or `trial_columns` inside the `pymovements` object.* The
  convention reuses the per-column object, and the `pymovements` object holds only the stamp.
- *`Inclusive` as the field name.* Too generic, it says nothing on a column that is not an
  offset column.
- *Arrow schema metadata in feather, or `Units` written to every feather sidecar.* The same fact
  would be stated twice and could drift.
- *Recomputing `Format` at every save.* Turns `label` into `string`.
- *An off switch for the sidecar.* The design starts strict and can relax later.
- *Schema version `1.0`.* pymovements itself is below `1.0.0`.
- *A warning when the sidecar and the definition disagree on the sampling rate.* It would fire
  per file on every dataset with mixed sampling rates.
- *The BIDS `Delimiter` field for list columns.* Loses the component names.

## Backwards compatibility

**csv needs a unit.** `Events.save` is new and raises on csv without a unit from its first
release. The two existing methods get the five-release window: `Dataset.save_events` and
`Gaze.save_events` will keep writing milliseconds to csv when no unit is specified, with a
`DeprecationWarning` from v0.30.0, and will raise from v0.35.0. csv writing itself is not
deprecated. Reading csv stays without an end date.

**Participants and Phenotype.** A `Format` entry that contradicts the dtype of its column is
written as it is today. From v0.30.0 save will replace it and warn. Their sidecars gain the
`pymovements` object, which is an additional top-level key for readers of these files.

**Provisional keys in v0.29.0.** The implementation of PMEP 1 writes its two entries as
`metadata['SamplingFrequency']` and `metadata['offset']['OffsetsInclusive']` and documents both
names as provisional. Accepting this PMEP makes them final. No file carries them before v0.30.0.

**What does not change.**

- feather output: the data file is written as before, with the sidecar beside it
- reading csv files written by earlier versions
- `Gaze.save` and its two YAML files

## Implementation

Target release is v0.30.0. One issue per line, drafted once the PMEP is accepted:

- [ ] sidecar reader and writer: the stamp, the version rules, the dict rules, the boundary
      rules
- [ ] `Events.save` and `Events.load`: the three formats, units, the raise on nested columns
- [ ] `Events.trial_columns` as a view on its metadata entry
- [ ] entry maintenance in `Events.drop`, `Events.unnest` and the methods that add or rename
      columns
- [ ] `verify_bids` for events: the five checks and the not-implemented notices
- [ ] `Dataset.save_events`, `Dataset.save`, `Dataset.load_event_files` and `Gaze.save_events`:
      delegation, `time_unit`, `verify_bids`, tsv, the csv deprecation
- [ ] `Participants` and `Phenotype`: the stamp, the entry without a column, the replaced
      `Format`
- [ ] changelog entry and documentation of the sidecar format

**Later applications.** Samples will get the sidecar through Recording and Session. Reading
measures and precomputed events will get it once they have save methods.

**Future work.**

- The default extension will move from feather to tsv through `extension=None` with a
  `DeprecationWarning`. This is blocked by automatic unnesting.
- Automatic unnesting of nested columns for text files comes with the struct columns of
  [#453](https://github.com/pymovements/pymovements/issues/453). Feather files change dtype
  with that refactor, which is a breaking schema version.
- Inheritance, where one sidecar applies to several data files.
- The BIDS layout with file naming, the mapping of `name` and the reference timestamp
  ([#1563](https://github.com/pymovements/pymovements/issues/1563)).

**Out of scope** are `Gaze` and its two YAML files, which Recording and Session supersede,
provenance chains and the BIDS `Sources` mapping, saved events as a resource definition, and
guards against direct changes to the dict.

# PMEP 2: Metadata sidecars for saved files

| | |
|---|---|
| **Status** | Draft |
| **Type** | Standards |
| **Author** | Daniel G. Krakowczyk |
| **Created** | 2026-10-01 |
| **Supersedes** | none |

## TL;DR

- Saved tabular files share one metadata sidecar, `<stem>.json` beside the data file: one dict
  per column, keyed by name, and file-level keys beside them.
- Each class adopts it in its own PMEP or issue, naming its `schema` value and the keys it
  writes.
- The metadata dict mirrors the sidecar. Load reads it as it is, save writes it back after
  filling missing `Format` and `Units`, replacing a `Format` that contradicts the frame with a
  warning.
- A top-level `pymovements` entry carries the pymovements stamp: `schema`, a label for the
  writing class, `schema_version`, initially `0.1.0`, and `version`, the writing package
  version.
- A BIDS-conformant file loads with `load(path)` alone, sidecar or not.
- `verify_bids` reports BIDS nonconformities: `'REQUIRED'` (the save default) and
  `'RECOMMENDED'` warn, `True` raises, `False` (the load default) is silent.
- `Participants` and `Phenotype` keep their released behavior and gain the pymovements stamp
  and `Sources`.

## What it looks like

Saving a participants table, before and after. The call does not change:

```python
participants = Participants(
    data=polars.DataFrame({'participant_id': ['sub-01', 'sub-02'], 'age': [23, 31]}),
    metadata={'age': {'Description': 'age of the participant', 'Units': 'years'}},
)
participants.save('participants.tsv')    # writes participants.tsv and participants.json
```

`participants.json` before (v0.28.0) holds the column entries, with the `Format` that the
constructor inferred:

```json
{
    "participant_id": {"Format": "string"},
    "age": {"Description": "age of the participant", "Units": "years", "Format": "integer"}
}
```

`participants.json` after (v0.30.0) carries the pymovements stamp in front of them:

```json
{
    "pymovements": {
        "schema": "participants",
        "schema_version": "0.1.0",
        "version": "0.30.0"
    },
    "participant_id": {"Format": "string"},
    "age": {"Description": "age of the participant", "Units": "years", "Format": "integer"}
}
```

The table in the code above was built from a frame, not read from a file, so there is no
`Sources` entry to write.

After `Participants.load('participants.tsv')` the metadata dict is the sidecar, pymovements stamp
included, and the file that was read is recorded as the source:

```python
participants.metadata
# {
#     'pymovements': {'schema': 'participants', 'schema_version': '0.1.0', 'version': '0.30.0'},
#     'participant_id': {'Format': 'string'},
#     'age': {'Description': 'age of the participant', 'Units': 'years', 'Format': 'integer'},
#     'Sources': ['/data/participants.tsv'],
# }
participants.data.schema['age']    # Int64, built from Format
```

`age` round-trips as an integer column because `Format` says `integer`, `Units` plays no part in
it.

Two rules show on the next save:

```python
participants.data = participants.data.cast({'age': polars.Float64})
participants.save('participants.tsv')
# UserWarning: Format 'integer' of column 'age' does not fit Float64, replaced by 'number'

participants.data = participants.data.drop('age')
participants.save('participants.tsv')
# UserWarning: metadata entry 'age' has a Format but the frame has no column 'age', kept
```

## Resulting signatures

This PMEP produces a file format, no new Python signature. The schema label and the schema
version are carried in the sidecar:

```text
<stem>.<extension>    the data file, any format the class writes
<stem>.json           the sidecar, a JSON object
```

The sidecar owns the `json` extension next to the stem, so no data format may use it.

| key | level | value |
|---|---|---|
| `pymovements` | top | the pymovements stamp, three keys |
| `schema` | stamp | label of the writing class, the lowercase class name |
| `schema_version` | stamp | version of the schema defined here, initially `0.1.0` |
| `version` | stamp | package version that wrote the file, provenance |
| `Sources` | top | BIDS provenance: list of the files directly used, as paths |
| `<column name>` | top | a dict that describes the column |
| `Format` | column | BIDS format: `string`, `number`, `integer`, `boolean`, `index`, `label` |
| `Units` | column | unit of the column, on a time column the unit it is written in |

**Adoption contract.** A class that adopts the sidecar carries two parameters in these roles,
whatever else its signatures hold:

```python
save(
    path: Path | str,
    *,
    metadata_path: Path | str | None = None,
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = 'REQUIRED',
    ...
) -> None

load(
    path: Path | str,
    metadata: Path | str | dict[str, Any] | None = None,
    *,
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = False,
    ...
) -> Self
```

`metadata_path` on save is the sidecar path, `<stem>.json` when None. `metadata` on load takes a
path or a dict.

## Motivation

`Participants` and `Phenotype` write a BIDS sidecar from their `metadata` dict, each on its own
terms. Nothing in the file says which schema it follows or which pymovements version wrote it. A
`Format` that contradicts the frame is written as it is, and an entry without a column is handled
by each class on its own.

More classes gain `save` and `load` methods as the data model grows. Without one definition each
would define its own sidecar, and the same rule would be stated several times and drift.

tsv and csv files store no dtypes. Today the caller of `load` has to pass the right parameters to
read such a file back as it was written. The sidecar is where that record belongs: `Format` and
`Units` say how a column is read.

## Specification

**Adoption.** The mechanism is what every sidecar shares: the shape, the pymovements stamp and
its version rules, the metadata dict rules, the Duration columns rule, the conflict rules and the
two required-level checks. It is defined once, here. A class adopts it in its own proposal or
issue by naming what belongs to that class: its `schema` value, the keys it writes beyond the
ones defined here, its own checks and its save keyword for the time unit. A class that has
adopted follows every rule below. Which class adopts next, and when, follows the
[data model roadmap](https://github.com/pymovements/pymovements/discussions/1763).

**The sidecar file.** A save writes `<stem>.json` next to the data file, or to `metadata_path`
when given. `metadata` on load takes a path or a dict. The sidecar holds one dict per column at
the top level, keyed by the column name, and the file-level keys beside them, with no wrapper
key around the columns. Two data files with the same stem share one sidecar path, which save
warns about, see Conflicts.

**The pymovements stamp.** The top-level `pymovements` entry holds exactly three keys. `schema`
is a stable identifier of the writing class, the lowercase class name, written from the first
version so that every file carries it once a generic loader dispatches on it. `schema_version`
is the version of the schema defined here, see below. `version` is the package version that
wrote the file, as provenance. Save generates the pymovements stamp and writes it as the first
key of the sidecar, the other keys keep the order of the metadata dict. Load reads
`schema_version` for the version check, at any position, and keeps the whole pymovements stamp
in the metadata dict. Save overwrites the entry at every save, so an edited pymovements stamp is
never written.

**The schema version** has three parts and is one version for the whole mechanism, initially
`0.1.0`. Its breaking and its additive part follow the package's own rule: below `1.0.0`
the minor is the breaking part and the patch the additive one, from `1.0.0` the major is
the breaking part and the minor the additive one. Schema `1.0.0` will be declared with
pymovements `1.0.0`. Load compares the pymovements stamp with the version it implements:

| pymovements stamp found | load |
|---|---|
| newer breaking part | refuses, the message names the version found |
| everything else | reads and keeps the keys it does not know |

A newer additive part does not warn. A sidecar without a pymovements stamp reads as the oldest
version, silently. This covers files written by pymovements before v0.30.0 and every third-party
BIDS sidecar: a breaking schema version never changes how a BIDS-defined key is read.

**The metadata dict** mirrors the sidecar. Three rules connect it to the file and the frame:

1. Load reads the sidecar into the metadata dict as it is, pymovements stamp included, and
   changes nothing in it. `Sources` is the one entry load does not take from the sidecar, see
   Sources.
2. For text files the loader builds the dtype of a column from `Format`. A column loads as a
   Duration only when `Format` is `number` or `integer` and `Units` is a time unit, see Duration
   columns.
3. Save writes the metadata dict as it is, after three changes: it fills `Format` and `Units`
   into column entries where they are missing, it replaces a `Format` that does not fit the
   dtype of its column, with a warning, and it writes the pymovements stamp. A Duration column
   is written as `number`. After save the metadata dict equals the file.

**Sources.** `Sources` is the BIDS provenance key with the BIDS meaning (BIDS derivatives,
file-level metadata): the files directly used in the creation of this file, one hop, not the
chain. Every adopting class carries it. A loaded object records the file that was read, as a
POSIX path, relative to the dataset root under a `Dataset` and absolute otherwise. A loaded
sidecar's own `Sources` entry is not carried into the metadata dict, since it describes that
file's provenance and not the loaded object's. A `Sources` entry given in `metadata=` is kept,
it states the caller's intent. Save writes the entry when the metadata dict holds it. BIDS
URIs as the value form come with the PMEP that writes a BIDS dataset, and the paths written
until then stay readable, a URI is a prefix on the same path.

**Format and Units.** `Format` says how the values of a column are read, `Units` states the
unit they are written in. Text files store no dtypes, so for them the loader builds the dtype
of a column from `Format`. pymovements writes six of the BIDS values, `integer`, `number`,
`index`, `boolean`, `string` and `label`, and reads each as one dtype, `label` as the same
dtype as `string`. Any other BIDS value, such as `date` or `uri`, sets no dtype: the column
keeps the dtype inferred from the data and the entry is kept. Save fills a missing `Format`
from the dtype and replaces one that contradicts it, see the metadata dict rules. `Units` sets
no dtype on its own. It refines the dtype where pymovements defines one for the unit, today
Duration alone, see Duration columns. Any other unit, such as `years`, `deg` or `px`, is read
and written as it is. A dtype for another unit extends this rule in its own proposal, the
sidecar shape stays.

**Duration columns.** A column loads as a Duration only when its `Format` is `number` or
`integer` and its `Units` is one of the symbols `s`, `ms`, `us` and `ns`, which BIDS
recommends over spelled-out names, written as a number. The one exception is a column whose
unit BIDS fixes: an adoption declares such columns, and they load as a Duration in that unit
when `Units` is absent. The unit used is written to `Units` at save as for every other Duration
column. A number column without a time unit in `Units` is otherwise never a Duration. `step`, a
sample index, carries no dtype here, the adoption that holds the sampling rate converts it. On
save to a text file the unit of a Duration column resolves in this order: a save keyword of the
adopting class, if it has one, then the `Units` entry of the column. Save raises when neither
gives a unit, and the unit used is written to `Units`, so no file ever holds a duration without
its unit. Columns with different `Units` entries are therefore written in different units.
Which unit an adopting class defaults to for its own time columns is the adoption's choice.

**Conflicts.**

| situation | outcome |
|---|---|
| keyword on `load` differs from its sidecar entry | keyword wins for the frame, the entry stays as read, warning names both values |
| `metadata=` given on `load` | replaces the sidecar |
| entry with a `Format` key and no such column | kept, warns at construction and at save |
| top-level key equals a column name, value is not a dict | load and save warn, kept as read, the column keeps its inferred dtype |
| data file with the same stem and another extension beside the target | save warns |
| direct change to the metadata dict or the frame | not checked until the next save |

The `Format` key is what makes an entry a column entry. Every other top-level dict, such as a free
user key, is file-level metadata and stays silent. The warnings for a column name that is not a
dict and for a same-stem data file do not depend on `verify_bids`, the message names the fix. Load
calls the constructor, so the warning for an entry without a column fires once per load.

**Verification.** `verify_bids` reports nonconformities at a BIDS level. `'REQUIRED'`, the
default on save, warns for each required-level finding. `'RECOMMENDED'` warns for
required-level and recommended-level findings. `True` raises on a required-level finding.
`False`, the default on load, is silent. The mechanism runs two required-level checks: the
data file is a tsv file with tab separators, and nulls are written as `n/a`. A csv file
therefore warns under the default, and the message names `verify_bids=False`. An adoption
declares its own checks, at either level, and the same `verify_bids` runs them.

**Loading BIDS files.** A BIDS-conformant data file loads with `load(path)` alone, whether or
not a sidecar exists next to it, since BIDS recommends the sidecar and does not require it.
The defaults on load are the BIDS conventions: a tsv file is read tab-separated, `n/a` is
read as null, a column takes its dtype from `Format` where pymovements maps it and from the
data otherwise, a column whose unit BIDS fixes loads as a Duration in that unit, see Duration
columns, and `verify_bids` is `False`. Without a sidecar the metadata dict holds only
`Sources`.

**Participants and Phenotype** already model the BIDS sidecar in `metadata`, and their
signatures and released behavior stay. They adopt the sidecar in this PMEP. `metadata` on load
takes a path or a dict, as today. `Participants.save` keeps its released default
`participants.json`. Four things change. Their
sidecars gain the pymovements stamp, with the `schema` values `participants` and `phenotype`.
They carry `Sources`. An entry with a `Format` key and no column warns and is kept, where today
both classes keep it silently. And a `Format` that contradicts the frame is replaced at save with
a warning. Neither class has a time column. `MeasurementToolMetadata` in a `Phenotype` sidecar is
file-level metadata and stays silent.

## Rationale

**Why the BIDS tabular shape.** BIDS conformity is the target of the design, and defaults should
conform. A user may deviate and gets a warning or a raise through verification. The BIDS common
principles (v1.11.1) let a data dictionary hold column fields "in addition to any other metadata
one wishes to include that describe the file as a whole", and they require: "If a field name
included in the data dictionary matches a column name in the TSV file, then that field MUST
contain a description of the corresponding column". The first sentence allows the file-level
keys and the `pymovements` entry. The second is the reason save raises on a key that equals a
column name without being a dict. A probe with bids-validator-deno 3.0.2 and 2.4.1, on a raw
and a derivative dataset with identical results, confirmed the shape:

| sidecar content | validator |
|---|---|
| top-level `pymovements` entry | clean |
| free user key | clean |
| `Sources` with a relative or absolute path, participants and events sidecars | clean |
| `Sources` not a list of strings, derivative dataset | error `JSON_SCHEMA_VALIDATION_ERROR` |
| extra field inside a column dict | clean |
| description for a nonexistent column | clean |
| `columns` wrapper around the column dicts | warning `TSV_ADDITIONAL_COLUMNS_UNDEFINED` |
| `Units: "ms"` on `onset` of a BIDS events file | warning `TSV_COLUMN_TYPE_REDEFINED` |

The validator inspects only the keys it knows, so clean means not looked at for the free key and
the `pymovements` entry. `Sources` it knows in a derivative dataset: the two controls show that
it checks the value to be a list of strings, and the clean rows show that it does not check the
form of the string. BIDS deprecates plain relative paths in `Sources` in favour of BIDS URIs,
which need a dataset root and `DatasetLinks`, see Future work.

**Why one definition and adoption per class.** A rule stated in several PMEPs drifts. The
mechanism is specified once, and a class that adopts it adds only what is its own: the schema
value and its keys. A later proposal cites this one instead of restating it.

**Why the metadata dict is the file.** Every variant in which load removed derived fields from the
metadata dict lost information and needed an exception to get it back: first `label`, which shares
its dtype with `string`, then the unit of a file that is loaded and saved again, then the
pymovements stamp, which would need an accessor of its own to stay inspectable. Nothing is removed,
so nothing needs restoring. `Sources` is the one exception on load, since the entry describes the
saved file's provenance and not that of the object reading it.

**Why entries without a column are kept.** pymovements never removes a metadata entry on its
own, and BIDS treats a description for a nonexistent column as other metadata. The `Format` key
is the line between the two: a column dict always carries one, file-level metadata never
does, so only an entry with `Format` can be a column that went missing.

**Why no default time unit.** A number without a unit is not a duration. Reading a bare number
column as seconds, or as any other unit, would be a silent factor on every value of a file that
meant something else, and there is no error to catch it. Writing always records the unit, so
the no-guess rule costs nothing on the files pymovements writes. What unit a BIDS events file
takes is the events adoption's rule, not the mechanism's.

**Alternatives rejected.**

- *A `columns` wrapper for the column dicts.* Not the BIDS shape, and the validator no longer
  finds the descriptions.
- *`Sources` inside the `pymovements` entry.* The `pymovements` entry holds only the pymovements
  stamp.
- *A schema version per class.* One version for the mechanism, since a class adds keys and
  the mechanism defines what a key means. A class that needs its own break adds its own
  strictness in its adoption.
- *A `file` key in the pymovements stamp naming the data file.* The clash of two data files on one
  sidecar path is caught at save, where it arises, and the key would be stale after a rename.
- *The pymovements stamp dropped from the metadata dict on load.* Inspection would need an accessor
  of its own, and the metadata dict would no longer be the file.
- *Arrow schema metadata in feather, or `Units` written to every feather sidecar.* The same fact
  would be stated twice and could drift.
- *Recomputing `Format` at every save.* Turns `label` into `string`.
- *Schema version `1.0`.* pymovements itself is below `1.0.0`.
- *The BIDS `Delimiter` field for list columns.* Loses the component names.

## Backwards compatibility

**Participants and Phenotype.** A `Format` entry that contradicts the dtype of its column is written
as it is today. From v0.30.0 save will replace it and warn. An entry with a `Format` key and no
column is kept silently today and will warn. Their sidecars gain the `pymovements` entry and
`Sources`, two additional top-level keys for readers of these files. Files written by earlier
versions carry no pymovements stamp and load as the oldest schema version, silently. A round trip
through save and load adds the `pymovements` entry to the metadata dict, where today it gives the
metadata dict back unchanged.

**What does not change.**

- the signatures and defaults of `Participants.save`, `Participants.load`, `Phenotype.save` and
  `Phenotype.load`
- `Gaze.save` and its two YAML files

## Implementation

Target release is v0.30.0. One issue per line, drafted once the PMEP is accepted:

- [ ] sidecar reader and writer: the pymovements stamp, the version rules, the metadata dict rules,
      the conflict rules
- [ ] `verify_bids`: the checks every adoption runs
- [ ] `Participants` and `Phenotype`: the pymovements stamp, `Sources`, the entry without a column,
      the replaced `Format`
- [ ] changelog entry and documentation of the sidecar format

**Later adoptions.** `Recording` adopts the sidecar for samples with a later PMEP on the
Recording and its files. `Events` adopts it with a later PMEP on the BIDS layout for events,
which brings `Events.save` and `Events.load`. Reading measures and precomputed events adopt it
once they have save methods.
Order and dates follow the
[data model roadmap](https://github.com/pymovements/pymovements/discussions/1763).

**Future work.**

- Feather files change dtype with the struct columns of
  [#453](https://github.com/pymovements/pymovements/issues/453), which is a breaking schema
  version.
- Inheritance, where one sidecar applies to several data files. It decides when a save writes
  no sidecar of its own.
- BIDS URIs as the value form of `Sources`. They need a dataset root and, across datasets,
  `DatasetLinks` in `dataset_description.json`, so they come with the PMEP that writes a BIDS
  dataset. The paths written until then stay readable, a URI is a prefix on the same path.

**Out of scope** are `Gaze` and its two YAML files, which Recording and Session supersede,
provenance chains beyond the one hop `Sources` records, guards against direct changes to the
metadata dict, `BIDSVersion` and `dataset_description.json`, which belong to the PMEP that writes a
BIDS dataset, and the generic loader that dispatches on `schema` together with its rule for a file
labeled for another class, which come with the PMEP that brings the loader.

# PMEP 2: Metadata sidecars for saved files

| | |
|---|---|
| **Status** | Draft |
| **Type** | Standards |
| **Author** | Daniel G. Krakowczyk |
| **Created** | 2026-10-01 |
| **Supersedes** | none |

Part of the [data model roadmap](https://github.com/pymovements/pymovements/discussions/1763).
The sidecar mechanism is defined here once and adopted per class. The BIDS layout for events
(expected PMEP 6, [#1563](https://github.com/pymovements/pymovements/issues/1563)) brings
`Events.save` and `Events.load` and adopts the sidecar there.

## TL;DR

- Saved tabular files will share one metadata sidecar: `<stem>.json` next to the data file, in
  the BIDS tabular shape, always written.
- The mechanism is defined once. Each class adopts it in its own PMEP or issue by naming its
  `schema` value, its initial schema version and the keys it writes. `Participants` and
  `Phenotype` adopt it here.
- The dict is the file. Load puts the sidecar into `metadata` as it is, save writes it back as
  it is. Where an entry contradicts the frame, the frame wins at save, with a warning.
- A `pymovements` object in the sidecar carries the schema version, initially `0.1.0`.
- tsv takes the BIDS defaults: tab, seconds, `n/a`. csv needs an explicit time unit. feather
  stays native.
- `verify_bids` reports nonconformities, as in `Phenotype`.
- Ships in v0.30.0. `Participants` and `Phenotype` keep their released behavior.

## What it looks like

Saving a participants table, before and after. The call does not change:

```python
participants = Participants(
    data=polars.DataFrame({'participant_id': ['sub-01', 'sub-02'], 'age': [23, 31]}),
    metadata={'age': {'Description': 'age of the participant', 'Units': 'years'}},
)
participants.save('participants.tsv')    # writes participants.tsv and participants.json
```

`participants.json` before (v0.28.0) holds the column objects, with the `Format` that the
constructor inferred:

```json
{
    "participant_id": {"Format": "string"},
    "age": {"Description": "age of the participant", "Units": "years", "Format": "integer"}
}
```

`participants.json` after (v0.30.0) carries the stamp beside them:

```json
{
    "participant_id": {"Format": "string"},
    "age": {"Description": "age of the participant", "Units": "years", "Format": "integer"},
    "pymovements": {
        "schema": "participants",
        "schema_version": "0.1.0",
        "version": "0.30.0",
        "file": "participants.tsv"
    }
}
```

**TODO (Daniel):** the decision record does not name the key that holds the data file name
inside the `pymovements` object. `file` is a placeholder. The `schema` value `participants` is a
placeholder as well, see the TODO in the Specification.

After `Participants.load('participants.tsv')` the dict is the sidecar, stamp included:

```python
participants.metadata
# {
#     'participant_id': {'Format': 'string'},
#     'age': {'Description': 'age of the participant', 'Units': 'years', 'Format': 'integer'},
#     'pymovements': {'schema': 'participants', 'schema_version': '0.1.0',
#                     'version': '0.30.0', 'file': 'participants.tsv'},
# }
participants.data.schema['age']    # Int64, built from Format
```

Two rules show on the next save. A `Format` that no longer fits its column, because the column
was cast in between, is replaced with a warning. An entry whose column was dropped from the frame
is kept and warns.

## Resulting signatures

`Participants` and `Phenotype` keep their signatures. The sidecar path is `metadata_path` on save
and `metadata`, a path or a dict, on load:

```python
Participants.save(path, *, verify_bids='REQUIRED', metadata_path='participants.json',
                  separator='\t', write_csv_kwargs=None, metadata_encoding='utf-8')
Participants.load(path, metadata=None, *, verify_bids=False, separator='\t', rename=None,
                  read_csv_kwargs=None, metadata_encoding='utf-8')

Phenotype.save(path, *, verify_bids='REQUIRED', metadata_path=None, separator='\t',
               write_csv_kwargs=None, metadata_encoding='utf-8')
Phenotype.load(path, metadata=None, *, separator='\t', rename=None, read_csv_kwargs=None,
               metadata_encoding='utf-8', verify_bids=False)
```

A class that adopts the sidecar later defines its own `save` and `load` with these two
parameters in the same roles.

**File structure.** The schema name and the schema version are carried in the sidecar:

```text
<stem>.tsv | <stem>.csv | <stem>.feather    the data file
<stem>.json                                 the sidecar, a JSON object
```

| key | level | value |
|---|---|---|
| `<column name>` | top | object that describes the column |
| `Format` | column | BIDS format: `string`, `number`, `integer`, `bool`, `index`, `label` |
| `Units` | column | unit of the column, on a time column the unit it is written in |
| `sources` | top | list of source files |
| `trial_columns` | top | list of trial column names |
| `pymovements` | top | the stamp: `schema`, `schema_version`, `version`, data file name |

`sources` and `trial_columns` are reserved for the classes that carry them. `sources` is written
as [#1655](https://github.com/pymovements/pymovements/pull/1655) defines it. Neither
`Participants` nor `Phenotype` writes either key. Any other key is kept as it is, on load and on
save.

**TODO (Daniel):** which value does `sources` hold after load? "The dict is the file" gives the
sidecar's value. The `add_source` docstring in
[#1655](https://github.com/pymovements/pymovements/pull/1655) says a loaded sidecar's `sources`
is not forwarded and the file that was read is recorded.

## Motivation

`Participants` and `Phenotype` already write a BIDS sidecar from their `metadata` dict, each on
its own terms. Nothing in the file says which schema the dict follows or which pymovements
version wrote it, so a reader cannot tell a file it can read from one it cannot. A `Format` that
contradicts the frame is written as it is. An entry without a column is handled by each class on
its own.

More classes will save files. The roadmap brings `save` and `load` for `Recording` and for
`Events`, and reading measures and precomputed events will follow. Without one definition each
class would define its own sidecar, and the same rule would be stated several times and drift.
Consistency across classes is preferred over a second format.

The text formats need the sidecar to round-trip. tsv and csv store no dtypes, so `Format` and
`Units` are the only record of how a column is to be read.

## Specification

**Adoption.** The mechanism is defined once, here. A class adopts it in its own PMEP or issue by
naming its `schema` value, its initial schema version and the keys it writes beyond the ones
defined here. `Participants` and `Phenotype` adopt it in this PMEP. A class that has adopted the
sidecar follows every rule below. The rules name a class only where that class deviates.

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

**The schema version** has three parts, and every schema has its own. Below `1.0.0` the minor is
the breaking position and the patch the additive one, as in the package. Schema `1.0.0` will be
declared with pymovements `1.0.0`. Load compares the stamp with the version it implements:

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
4. On a class that has `trial_columns`, the attribute is a view on `metadata['trial_columns']`.
   The constructor keyword writes the entry. A keyword that differs from an entry in `metadata=`
   raises.

Assigning a whole new dict to `metadata` also drops the trial columns, and nothing raises
afterwards.

**TODO (Daniel):** does save write what it fills in, replaces and stamps back into the dict in
memory, or only into the file? The decision record says the stamp is "overwritten at every
save".

**Units.** `Units` on a time column states the unit the column is written in as a number. A
column whose `Units` is a time unit loads as a Duration. On save to a text file the unit of a
time column resolves in this order: a per-call keyword, where the adopting class defines one,
then the `Units` entry of the column, then the default of the format. Columns with different
entries are therefore written in different units.

**Boundaries.**

| situation | outcome |
|---|---|
| keyword on `load` differs from its sidecar entry | keyword wins, warning names both values |
| `metadata=` given on `load` | replaces the sidecar |
| column entry without its column | kept, warning at construction, load and save |
| top-level key equals a column name, value is not an object | save raises, load warns |
| direct change to the dict or the frame | not checked until the next save |

The message of the raise names the fix. The `trial_columns` setter validates its value as the
constructor does, and save applies the same check.

**TODO (Daniel):** how is a column entry without its column told apart from a file-level key
whose value is an object, such as `MeasurementToolMetadata` in a `Phenotype` sidecar or a free
user key?

**Formats.** The extension of the path selects the format:

| | separator | time unit |
|---|---|---|
| tsv | tab | seconds by default |
| csv | comma | must be specified |
| feather | none | native Duration |

tsv takes the BIDS defaults: tab, seconds and `n/a` for nulls. csv has no default unit. The unit
comes from the `Units` entries or from the per-call keyword of the adopting class, and save
raises without one.

**Nested columns** raise for text files. Feather stores them natively. The text format is
thereby specified on flat columns only, independent of how nested columns are stored.

**Verification.** `verify_bids` works as in `Phenotype`: `'REQUIRED'`, the default on save,
warns for each finding, `True` raises and `False` is silent. The mechanism defines the checks
that every adoption runs:

- a top-level key that matches a column name is an object
- nulls are written as `n/a`
- the separator is a tab

A class adds its own checks in its adoption.

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
| description for a nonexistent column | clean |
| `columns` wrapper around the column objects | warning `TSV_ADDITIONAL_COLUMNS_UNDEFINED` |
| `Units: "ms"` on `onset` of a BIDS events file | warning `TSV_COLUMN_TYPE_REDEFINED` |

The validator inspects only the keys it knows, so clean means not looked at.

**Why one definition and adoption per class.** A rule stated in several PMEPs drifts. The
mechanism is specified once, and a class that adopts it adds only what is its own: the schema
value, the version and its keys. A later proposal cites this one instead of restating it.

**Why the dict is the file.** Every variant in which load removed derived fields from the dict
lost information and needed an exception to get it back: first `label`, which shares its dtype
with `string`, then the unit of a file that is loaded and saved again. Nothing is removed, so
nothing needs restoring.

**Why entries without a column are kept.** pymovements never removes a metadata entry on its
own, and BIDS treats a description for a nonexistent column as other metadata.

**Why csv has no default unit.** Today's csv files written by pymovements hold milliseconds. A
seconds default would change every number by a factor of 1000 for external readers of these
files, without an error. tsv has no such history, and BIDS requires seconds there.

**Alternatives rejected.**

- *A `columns` wrapper for the column objects.* Not the BIDS shape, and the validator no longer
  finds the descriptions.
- *`sources` or `trial_columns` inside the `pymovements` object.* The `pymovements` object holds
  only the stamp.
- *Arrow schema metadata in feather, or `Units` written to every feather sidecar.* The same fact
  would be stated twice and could drift.
- *Recomputing `Format` at every save.* Turns `label` into `string`.
- *An off switch for the sidecar.* The design starts strict and can relax later.
- *Schema version `1.0`.* pymovements itself is below `1.0.0`.
- *The BIDS `Delimiter` field for list columns.* Loses the component names.

## Backwards compatibility

**Participants and Phenotype.** A `Format` entry that contradicts the dtype of its column is
written as it is today. From v0.30.0 save will replace it and warn. Their sidecars gain the
`pymovements` object, which is an additional top-level key for readers of these files. Files
written by earlier versions carry no stamp and load under the no-stamp rule.

**What does not change.**

- the signatures and defaults of `Participants.save`, `Participants.load`, `Phenotype.save` and
  `Phenotype.load`
- `Gaze.save` and its two YAML files

## Implementation

Target release is v0.30.0. One issue per line, drafted once the PMEP is accepted:

- [ ] sidecar reader and writer: the stamp, the version rules, the dict rules, the boundary
      rules
- [ ] `verify_bids`: the checks every adoption runs
- [ ] `Participants` and `Phenotype`: the stamp, the entry without a column, the replaced
      `Format`
- [ ] changelog entry and documentation of the sidecar format

**Later adoptions.** `Recording` adopts the sidecar for samples with the Recording and its files
PMEP (expected PMEP 5). `Events` adopts it with the BIDS layout for events (expected PMEP 6,
[#1563](https://github.com/pymovements/pymovements/issues/1563)), which brings `Events.save` and
`Events.load`. Reading measures and precomputed events adopt it once they have save methods.
Numbers and dates follow the
[data model roadmap](https://github.com/pymovements/pymovements/discussions/1763).

**Future work.**

- Automatic unnesting of nested columns for text files comes with the struct columns of
  [#453](https://github.com/pymovements/pymovements/issues/453). Feather files change dtype
  with that refactor, which is a breaking schema version.
- Inheritance, where one sidecar applies to several data files.

**Out of scope** are `Gaze` and its two YAML files, which Recording and Session supersede,
provenance chains and the BIDS `Sources` mapping, and guards against direct changes to the
dict.

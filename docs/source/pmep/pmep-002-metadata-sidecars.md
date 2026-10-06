# PMEP 2: Metadata sidecars for saved files

| | |
|---|---|
| **Status** | Draft |
| **Type** | Standards |
| **Author** | Daniel G. Krakowczyk |
| **Created** | 2026-10-01 |
| **Supersedes** | none |

Part of the [data model roadmap](https://github.com/pymovements/pymovements/discussions/1763).
The sidecar mechanism is defined here once and adopted per class. A later PMEP on the BIDS
layout for events brings `Events.save` and `Events.load` and adopts the sidecar there.

## TL;DR

- Saved tabular files will share one metadata sidecar: `<stem>.json` next to the data file, in
  the BIDS tabular shape, always written.
- The mechanism is defined once. Each class adopts it in its own PMEP or issue by naming its
  `schema` value and the keys it writes. `Participants` and `Phenotype` adopt it here.
- The dict is the file. Load puts the sidecar into `metadata` as it is, stamp included, save
  writes it back as it is. Save fills `Format` and `Units` into the dict where they are missing,
  and where an entry contradicts the frame, the frame wins at save, with a warning.
- A `pymovements` object in the sidecar carries the stamp: `schema`, a label for the writing
  class, `schema_version`, initially `0.1.0`, and `version`, the package version as provenance.
- A time column is a Duration only with a time unit in `Units`. Nothing is guessed, on load or
  on save.
- `verify_bids` reports nonconformities, as in `Phenotype`.
- Ships in v0.30.0. `Participants` and `Phenotype` keep their released behavior and gain the
  stamp and `sources`.

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

`participants.json` after (v0.30.0) carries the stamp in front of them:

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

The table was built from a frame, not read from a file, so there is no `sources` entry to write.

After `Participants.load('participants.tsv')` the dict is the sidecar, stamp included, and the
file that was read is recorded as the source:

```python
participants.metadata
# {
#     'pymovements': {'schema': 'participants', 'schema_version': '0.1.0', 'version': '0.30.0'},
#     'participant_id': {'Format': 'string'},
#     'age': {'Description': 'age of the participant', 'Units': 'years', 'Format': 'integer'},
#     'sources': ['/data/participants.tsv'],
# }
participants.data.schema['age']    # Int64, built from Format
```

Two rules show on the next save. A `Format` that no longer fits its column, because the column
was cast in between, is replaced with a warning. An entry whose column was dropped from the frame
still carries its `Format`, so it is kept and warns.

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
| `<column name>` | top | object that describes the column |
| `Format` | column | BIDS format: `string`, `number`, `integer`, `bool`, `index`, `label` |
| `Units` | column | unit of the column, on a time column the unit it is written in |
| `sources` | top | list of source files |
| `pymovements` | top | the stamp: `schema`, `schema_version`, `version` |

`sources` is a key of the mechanism, carried by every class as
[#1655](https://github.com/pymovements/pymovements/pull/1655) defines it: a loaded object records
the file that was read, and save writes the entry when the dict holds it. A loaded sidecar's own
`sources` entry is not carried into the dict, see The dict below. Any other key is kept as it is,
on load and on save.

**Adoption contract.** A class that adopts the sidecar carries two parameters in these roles,
whatever else its signatures hold:

```python
save(path, *, metadata_path=None, verify_bids='REQUIRED', ...)
load(path, metadata=None, *, verify_bids=False, ...)
```

`metadata_path` gives the sidecar a custom path on save, `metadata` takes a path or a dict on
load. `Participants` and `Phenotype` already carry both, and their signatures stay as released.

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
naming its `schema` value and the keys it writes beyond the ones defined here. `Participants` and
`Phenotype` adopt it in this PMEP. A class that has adopted the sidecar follows every rule below.
The rules name a class only where that class deviates.

**The sidecar file.** Every save will write `<stem>.json` next to the data file. There is no
switch to turn it off. `metadata_path` on save gives a custom path, and `metadata` on load takes
a path or a dict, as in `Phenotype` today. `Participants.save` keeps its released default
`participants.json`. The sidecar has the BIDS tabular shape: one object per column at the top
level, keyed by the column name, and the file-level keys beside them. There is no wrapper object
around the columns. Two data files with the same stem share one sidecar path, which save warns
about, see Boundaries.

**The stamp.** The top-level `pymovements` object holds only the stamp, three fields. `schema`
is a stable identifier of the writing class. A future generic loader dispatches on it to find
the class that reads the file, and the mapping from the value to the class is that loader's. The
value is the lowercase class name today, so a class rename is a mapping entry and not a schema
change. The value is written now because a key can be added to the schema later but not to files
already on disk. In this PMEP no `load` reads `schema`, and what happens when a class is asked
to load a file labeled for another class is left to the PMEP that brings the loader.
`schema_version` is the version of the schema defined here, see below. `version` is the package
version that wrote the file, provenance only, and no `load` reads it. Save generates the stamp
and writes it as the first key of the sidecar, so a reader sees it before the columns. The other
keys keep the order of the dict. Load reads the stamp for the version check, at any position,
and keeps it in the dict, where it can be inspected. Save overwrites the entry at every save, so
an edited stamp is never written.

**The schema version** has three parts and is one version for the whole mechanism, initially
`0.1.0`. Its breaking and its additive position follow the package's own rule: below `1.0.0`
the minor is the breaking position and the patch the additive one, from `1.0.0` the major is
the breaking position and the minor the additive one. Schema `1.0.0` will be declared with
pymovements `1.0.0`. Load compares the stamp with the version it implements:

| stamp found | load |
|---|---|
| newer breaking position | refuses, the message names the version found |
| everything else | reads and keeps the keys it does not know |

A newer additive position does not warn. A sidecar without a stamp reads as the oldest version,
silently. This covers the released `Participants` and `Phenotype` sidecars and every third-party
BIDS sidecar. An adopting class may add its own strictness in its adoption, this PMEP adds
none.

**The dict.** Three rules connect `metadata`, the sidecar and the frame:

1. The dict is the file. Load puts the sidecar into the dict as it is, stamp included, and save
   writes the dict as it is. The one exception is `sources`: a loaded sidecar's `sources` entry
   is not carried into the dict, the file that was read becomes the source, as
   [#1655](https://github.com/pymovements/pymovements/pull/1655) defines it.
2. For text files the loader builds the dtype of a column from `Format`, and the writer fills
   `Format` and `Units` into the dict where they are missing. A column loads as a Duration only
   when `Format` is `number` or `integer` and `Units` is a time unit, see Units. A Duration
   column is written as `number`. Feather needs neither. The writer never adds `Units` to a
   feather sidecar, and an entry that is already in the dict is carried.
3. On contradiction the frame wins. A `Format` that does not fit the dtype of its column is
   replaced in the dict at save, with a warning. Save writes the stamp into the dict as well.
   After save the dict equals the file.

**Units.** `Units` is descriptive, with one exception. A column loads as a Duration only when
its `Format` is `number` or `integer` and its `Units` is one of `s`, `ms`, `us` and `ns`, and
`Units` then states the unit the column is written in as a number. Every other unit, such as
`years`, `deg` or `px`, changes no dtype and round-trips untouched. A number column without a
time unit in `Units` is never read as a Duration, whatever its name, since a number without a
unit is not a duration. On save to a text file the unit of a Duration column resolves in this
order: a per-call keyword, where the adopting class defines one, then the `Units` entry of the
column. Save raises when neither gives a unit, and the unit used is written to `Units`, so no
file ever holds a duration without its unit. Columns with different entries are therefore
written in different units. Which unit an adopting class defaults to for its own time columns is
the adoption's choice. `Participants` and `Phenotype` have no time columns.

**Boundaries.**

| situation | outcome |
|---|---|
| keyword on `load` differs from its sidecar entry | keyword wins, warning names both values |
| `metadata=` given on `load` | replaces the sidecar |
| object entry with a `Format` key and no such column | kept, warns at construction, load and save |
| top-level key equals a column name, value is not an object | save raises, load warns |
| data file with the same stem and another extension beside the target | save warns |
| direct change to the dict or the frame | not checked until the next save |

The `Format` key is what makes an entry a column entry. Every other top-level object, such as
`MeasurementToolMetadata` in a `Phenotype` sidecar or a free user key, is file-level metadata
and stays silent. The raise on a key that equals a column name does not depend on `verify_bids`,
and its message names the fix. The same-stem warning does not depend on `verify_bids` either.

**Formats.** The extension of the path selects the format. tsv, csv and feather are the formats
the classes write today, and a format added later takes the same sidecar. The format gives the
default separator, and `separator=` overrides it on every class. Feather stores dtypes natively
and needs neither `Format` nor `Units` to round-trip. What else a data file format holds, nulls,
nested columns, a default time unit, is the data file's concern and not the sidecar's.

**Verification.** `verify_bids` works as in `Phenotype`: `'REQUIRED'`, the default on save,
warns for each finding, `True` raises and `False` is silent. The mechanism defines the checks
that every adoption runs:

- nulls are written as `n/a`
- the separator is a tab

A class adds its own checks in its adoption.

**Participants and Phenotype** already model the BIDS sidecar in `metadata`, and their
signatures and released behavior stay. Four things change. Their sidecars gain the stamp, with
the `schema` values `participants` and `phenotype`. They carry `sources`. An object entry with a
`Format` key and no column warns and is kept, where today both classes keep it silently. And a
`Format` that contradicts the frame is replaced at save with a warning.

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
value and its keys. A later proposal cites this one instead of restating it.

**Why the dict is the file.** Every variant in which load removed derived fields from the dict
lost information and needed an exception to get it back: first `label`, which shares its dtype
with `string`, then the unit of a file that is loaded and saved again, then the stamp, which
would need an accessor of its own to stay inspectable. Nothing is removed, so nothing needs
restoring. `sources` is the one exception on load, since the entry describes the saved file's
provenance and not that of the object reading it.

**Why entries without a column are kept.** pymovements never removes a metadata entry on its
own, and BIDS treats a description for a nonexistent column as other metadata. The `Format` key
is the line between the two: a column object always carries one, file-level metadata never
does, so only an entry with `Format` can be a column that went missing.

**Why no default time unit.** A number without a unit is not a duration. Reading a bare number
column as seconds, or as any other unit, would be a silent factor on every value of a file that
meant something else, and there is no error to catch it. Writing always records the unit, so
the no-guess rule costs nothing on the files pymovements writes. What unit a BIDS events file
takes is the events adoption's rule, not the mechanism's.

**Alternatives rejected.**

- *A `columns` wrapper for the column objects.* Not the BIDS shape, and the validator no longer
  finds the descriptions.
- *`sources` inside the `pymovements` object.* The `pymovements` object holds only the stamp.
- *A schema version per class.* One version for the mechanism, since a class adds keys and
  the mechanism defines what a key means. A class that needs its own break adds its own
  strictness in its adoption.
- *A `file` key in the stamp naming the data file.* The clash of two data files on one sidecar
  path is caught at save, where it arises, and the key would be stale after a rename.
- *The stamp dropped from the dict on load.* Inspection would need an accessor of its own, and
  the dict would no longer be the file.
- *Arrow schema metadata in feather, or `Units` written to every feather sidecar.* The same fact
  would be stated twice and could drift.
- *Recomputing `Format` at every save.* Turns `label` into `string`.
- *An off switch for the sidecar.* The design starts strict and can relax later.
- *Schema version `1.0`.* pymovements itself is below `1.0.0`.
- *The BIDS `Delimiter` field for list columns.* Loses the component names.

## Backwards compatibility

**Participants and Phenotype.** A `Format` entry that contradicts the dtype of its column is
written as it is today. From v0.30.0 save will replace it and warn. An object entry with a
`Format` key and no column is kept silently today and will warn. Their sidecars gain the
`pymovements` object and `sources`, two additional top-level keys for readers of these files.
Files written by earlier versions carry no stamp and load as the oldest schema version, silently.
A round trip through save and load adds the `pymovements` entry to the dict, where today it
gives the dict back unchanged.

**What does not change.**

- the signatures and defaults of `Participants.save`, `Participants.load`, `Phenotype.save` and
  `Phenotype.load`
- `Gaze.save` and its two YAML files

## Implementation

Target release is v0.30.0. One issue per line, drafted once the PMEP is accepted:

- [ ] sidecar reader and writer: the stamp, the version rules, the dict rules, the boundary
      rules
- [ ] `verify_bids`: the checks every adoption runs
- [ ] `Participants` and `Phenotype`: the stamp, `sources`, the entry without a column, the
      replaced `Format`
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
- Inheritance, where one sidecar applies to several data files.

**Out of scope** are `Gaze` and its two YAML files, which Recording and Session supersede,
provenance chains and the BIDS `Sources` mapping, guards against direct changes to the dict,
`BIDSVersion` and `dataset_description.json`, which belong to the PMEP that writes a BIDS
dataset, and the generic loader that dispatches on `schema` together with its rule for a file
labeled for another class, which come with the PMEP that brings the loader.

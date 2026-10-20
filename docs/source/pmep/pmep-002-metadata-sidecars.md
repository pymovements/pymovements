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
- `metadata` is the memory, `sidecar` the file as read or written, equal after save.
  pymovements fills a missing entry silently and replaces a contradicting one with a warning.
- A top-level `pymovements` entry, written at construction, carries the pymovements stamp.
  `schema` labels the writing class, `schema_version` starts at `0.1.0` and `version` names the
  writing package version.
- A BIDS-conformant file loads with the class's `load(path)` alone, sidecar or not.
- `verify_bids` reports BIDS nonconformities and is off by default: `'REQUIRED'` and
  `'RECOMMENDED'` warn, `True` raises, `False` is silent.
- `Participants` and `Phenotype` adopt the sidecar here: they gain the pymovements stamp,
  `Sources` and `sidecar`, and verification on save becomes opt-in.

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

The data object in the code above was built from a frame, not read from a file, so there is no
`Sources` entry to write.

After `Participants.load('participants.tsv')` under a later pymovements, say v0.31.0, `sidecar`
holds the file as read. `metadata` holds what describes the loaded data object: the file's
column entries, the running pymovements stamp, and the file that was read as the source:

```python
participants.sidecar
# {
#     'pymovements': {'schema': 'participants', 'schema_version': '0.1.0', 'version': '0.30.0'},
#     'participant_id': {'Format': 'string'},
#     'age': {'Description': 'age of the participant', 'Units': 'years', 'Format': 'integer'},
# }
participants.metadata
# {
#     'pymovements': {'schema': 'participants', 'schema_version': '0.1.0', 'version': '0.31.0'},
#     'participant_id': {'Format': 'string'},
#     'age': {'Description': 'age of the participant', 'Units': 'years', 'Format': 'integer'},
#     'Sources': ['/data/participants.tsv'],
# }
participants.data.schema['age']    # Int64, built from Format
```

The two pymovements stamps differ in `version`: `sidecar` carries the one that wrote the file,
`metadata` the running one. `age` round-trips as an integer column because `Format` says
`integer`. `Units` plays no part in it.

On the next save a contradicting `Format` is replaced and an entry without a column stays, and after
each save `sidecar` equals `metadata` again. The saves go over the loaded file, so they say
`overwrite=True`, see <a href="#conflicts">Conflicts</a>:

```python
participants.data = participants.data.cast({'age': polars.Float64})
participants.save('participants.tsv', overwrite=True)
# UserWarning: Format 'integer' of column 'age' does not fit Float64, replaced by 'number'

participants.data = participants.data.drop('age')
participants.save('participants.tsv', overwrite=True)
# no warning, the entry 'age' stays, see Conflicts
```

## Resulting signatures

This PMEP produces a file format and one read-only attribute, `sidecar`, on every adopting
class. The schema label and the schema version are carried in the sidecar:

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
| `Sources` | top | BIDS provenance: list of the files directly used, as paths <a href="#references">[1]</a> |
| `<column name>` | top | a dict that describes the column |
| `Format` | column | BIDS format: `string`, `number`, `integer`, `boolean`, `index`, `label` <a href="#references">[2]</a> |
| `Units` | column | unit of the column's values <a href="#references">[2]</a>, on a Duration column also how it loads, see <a href="#duration-columns">Duration columns</a> |

### Adoption contract

A class that adopts the sidecar carries two attributes and two parameters in these roles, whatever
else its signatures hold:

```python
metadata: dict[str, Any]          # what describes the data object, see The metadata dict
sidecar: dict[str, Any] | None    # the file as read by load or written by save, kept by clone
                                  # None without a file: after split, concat or a constructor call
                                  # read-only, changes to what it returns do not reach it

save(
    path: Path | str,
    *,
    sidecar_path: Path | str | None = None,   # <stem>.json when None
    sidecar_encoding: str = 'utf-8',          # encoding of the sidecar file
    overwrite: bool = False,                  # write over a file at path or at the sidecar path
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = False,   # released shape, see Scope
    ...
) -> None

load(
    path: Path | str,
    metadata: dict[str, Any] | None = None,
    *,
    sidecar_path: Path | str | None = None,   # <stem>.json when None
    sidecar_encoding: str = 'utf-8',          # encoding of the sidecar file
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = False,   # released shape, see Scope
    ...
) -> Self
```

### Participants and Phenotype signatures

`save` and `load` on both classes: before, with the deprecated keywords from v0.30.0, and after
their removal in v0.35.0. `...` stands for the keywords that do not change. The two classes
differ only in the released `metadata_path` default:

```python
# before, v0.28.0
def save(
    self,
    path: Path | str,
    *,
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = 'REQUIRED',
    metadata_path: Path | str | None = None,    # 'participants.json' on Participants
    metadata_encoding: str = 'utf-8',
    ...
) -> None

def load(
    path: Path | str,
    metadata: Path | str | dict[str, Any] | None = None,
    *,
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = False,
    metadata_encoding: str = 'utf-8',
    ...
) -> Self

# from v0.30.0, the deprecated keywords warn and are removed in v0.35.0
def save(
    self,
    path: Path | str,
    *,
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = False,
    sidecar_path: Path | str | None = None,
    sidecar_encoding: str = 'utf-8',
    overwrite: bool = False,
    metadata_path: Path | str | None = None,    # deprecated, use sidecar_path
    metadata_encoding: str | None = None,       # deprecated, use sidecar_encoding
    ...
) -> None

def load(
    path: Path | str,
    metadata: Path | str | dict[str, Any] | None = None,   # a path is deprecated, use sidecar_path
    *,
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = False,
    sidecar_path: Path | str | None = None,
    sidecar_encoding: str = 'utf-8',
    metadata_encoding: str | None = None,       # deprecated, use sidecar_encoding
    ...
) -> Self

# from v0.35.0
def save(
    self,
    path: Path | str,
    *,
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = False,
    sidecar_path: Path | str | None = None,
    sidecar_encoding: str = 'utf-8',
    overwrite: bool = False,
    ...
) -> None

def load(
    path: Path | str,
    metadata: dict[str, Any] | None = None,
    *,
    verify_bids: Literal['REQUIRED', 'RECOMMENDED'] | bool = False,
    sidecar_path: Path | str | None = None,
    sidecar_encoding: str = 'utf-8',
    ...
) -> Self
```

## Motivation

A saved file should load back as it was saved. tsv and csv files store no dtypes, so a text file
alone does not say whether a column holds integers, labels or durations, nor which unit a time
column is written in. Today the caller of `load` has to know that and pass the right parameters, and
a wrong guess changes the data silently: a duration read as a plain number, a label column of digits
read as integers. The sidecar is where that record belongs. BIDS defines it for this purpose
<a href="#references">[2]</a>, `Format` and `Units` say how a column is read, and a file that
carries its record loads with the path alone.

Today `Participants` and `Phenotype` write a BIDS sidecar from their `metadata` dict, each on its
own terms, and `Events` and more classes gain `save` and `load` as the data model grows. Nothing
in the files says which schema they follow or which pymovements version wrote them. A `Format`
that contradicts the frame is written as it is, and an entry without a column is handled by each
class on its own. Without one definition every class would define its own sidecar, and the same
rule would be stated several times and drift.

## Specification

### Adoption

The mechanism is what every sidecar shares: the shape, the pymovements stamp and its version rules,
the two attributes and the metadata dict rules, the Duration columns rule, the conflict rules and
the two required-level checks. It is defined once, here. A class adopts it in its own proposal or
issue by naming what belongs to that class: its `schema` value, the keys it writes beyond the ones
defined here, its own checks and its save keyword for the time unit. The keys an adoption adds are
top-level keys beside `Sources`, never part of the pymovements stamp. A class that has adopted
follows every rule below.

### The sidecar file

A save writes `<stem>.json` next to the data file, or to `sidecar_path` when given. The stem is the
file name without its format extension, a compression suffix included: `sub-01_physio.tsv.gz` has
the sidecar `sub-01_physio.json` and `data.v2.tsv` has `data.v2.json`. A load reads it from the same
place. The sidecar holds one dict per column at the top level, keyed by the column name, and the
file-level keys beside them, with no wrapper key around the columns. Save writes and load reads the
sidecar in `sidecar_encoding`, `utf-8` by default. The `sidecar` attribute holds the file as read by
load or written by save, the `metadata` attribute what describes the data object, see
<a href="#the-metadata-dict">The metadata dict</a>.

### The pymovements stamp

The top-level `pymovements` entry holds exactly three keys. `schema` is a stable identifier of the
writing class, the lowercase class name. `schema_version` is the version of the schema defined here,
see <a href="#the-schema-version">The schema version</a>. `version` is the package version that
wrote the file, as provenance. Construction writes the running pymovements stamp into the metadata
dict. Save writes it as the first key of the sidecar, the other keys keep the order of the metadata
dict. Load reads `schema_version` on `sidecar` for the version check, at any position, and the
file's pymovements stamp stays there. A pymovements stamp in the metadata dict that differs from the
running one is replaced with a warning, see <a href="#the-metadata-dict">The metadata dict</a>, so
an edited pymovements stamp is never written.

### The schema version

The schema version has three parts and is one version for the whole mechanism, initially `0.1.0`.
Its breaking and its additive part follow the package's own rule. Below `1.0.0` the minor is the
breaking part and the patch the additive one. From `1.0.0` the major is the breaking part and the
minor the additive one. Schema `1.0.0` will be declared with pymovements `1.0.0`. Load compares the
pymovements stamp on `sidecar` with the version it implements:

| pymovements stamp found | load |
|---|---|
| newer breaking part | refuses, the message names the version found |
| everything else | reads and keeps the keys it does not know |

A newer additive part does not warn. A sidecar without a pymovements stamp reads as the oldest
version, silently. This covers files written by pymovements before v0.30.0 and every third-party
BIDS sidecar: a breaking schema version never changes how a BIDS-defined key is read.

### The metadata dict

`metadata` holds what describes the data object, `sidecar` the file. Three rules connect them to
each other and to the frame:

1. Load reads the sidecar into `sidecar` as it is and builds the metadata dict from it with three
   changes. The file's own `Sources` is left out and the loaded object's added, see
   <a href="#sources">Sources</a>. The file's pymovements stamp is left out. An entry a load keyword
   overrides is replaced by the keyword's value, with a warning. `sidecar` keeps every value as
   read.
2. For text files the loader builds the dtype of a column from `Format`. A column loads as a
   Duration when `Format` is `number` or `integer` and `Units` is a time unit, or when BIDS fixes
   its unit, see <a href="#duration-columns">Duration columns</a>.
3. Save writes the metadata dict to the sidecar path after the corrections below, as indented JSON
   with one key per line at every depth, then sets `sidecar` to what it wrote. A Duration column is
   written as `number`, see <a href="#duration-columns">Duration columns</a>. After save the
   metadata dict, `sidecar` and the file are equal.

One rule covers every correction pymovements makes to the metadata dict: it fills a missing
entry silently and replaces an entry that contradicts the frame or the file with a warning that
names both values. It runs at construction and at save:

| when | what | outcome |
|---|---|---|
| construction | `Format` missing | filled from the dtype of its column, silently |
| construction | pymovements stamp missing | the running one written, silently |
| save | `Format` missing | filled from the dtype of its column, silently |
| save | `Units` missing on a Duration column | filled from the unit save writes, silently |
| construction and save | `Format` does not fit the dtype of its column | replaced, the warning names both values |
| construction and save | the dtype of a column maps to no `Format` value, such as Date, Datetime, List or Struct | raises, as today, the message names the column and its dtype |
| construction and save | pymovements stamp differs from the running one | replaced, the warning names both values, an equal one is silent |

The load keyword in rule 1 is the same rule on the file side: the keyword states what the file
contains and the entry contradicts it. The save keyword of an adopting class for the time unit is no
correction but a conversion the caller chose. It wins silently and the unit written goes into
`Units`, see <a href="#duration-columns">Duration columns</a>.

### Sources

`Sources` is the BIDS provenance key with the BIDS meaning <a href="#references">[1]</a>: the files
directly used in the creation of this file, one hop, not the chain. Every adopting class carries it.
A loaded object records the file that was read, as a POSIX path, relative to the dataset root under
a `Dataset` and absolute otherwise. A loaded sidecar's own `Sources` entry stays on `sidecar` and is
not carried into the metadata dict. It describes that file's provenance and not the loaded object's.
A `Sources` entry given in `metadata=` is kept, it states the caller's intent. Save writes the entry
when the metadata dict holds it. A save over the loaded file therefore writes that file's path as
its own source, the record of a round trip. pymovements output belongs in a derivatives directory,
see <a href="#why-an-existing-file-raises">Why an existing file raises</a>. The paths written stay
readable under a BIDS URI, a URI is a prefix on the same path.

### Format and Units

`Format` says how the values of a column are read, `Units` states the unit they are written in. Text
files store no dtypes, so for them the loader builds the dtype of a column from `Format`.
pymovements writes six of the BIDS values <a href="#references">[2]</a>: `integer`, `number`,
`index`, `boolean`, `string` and `label`. It reads each as one dtype, `label` as the same dtype as
`string`. Any other BIDS value, such as `date` or `uri`, sets no dtype: the column keeps the dtype
inferred from the data and the entry is kept. Save fills a missing `Format` from the dtype and
replaces one that contradicts it, see <a href="#the-metadata-dict">The metadata dict</a>. `Units`
sets no dtype on its own. It refines the dtype where pymovements defines one for the unit, today
Duration alone, see <a href="#duration-columns">Duration columns</a>. Any other unit, such as
`years`, `deg` or `px`, is read and written as it is.

### Duration columns

A column loads as a Duration only when its `Format` is `number` or `integer` and its `Units` is one
of the symbols `s`, `ms`, `us` and `ns`, written as a number. BIDS recommends these symbols over
spelled-out names <a href="#references">[3]</a>. `Units` names the unit in the file. Load converts
the values into the Duration unit the adopting class uses in memory, and save converts them back
into the unit it writes. The one exception is a column whose unit BIDS fixes: an adoption declares
such columns, and they load as a Duration from that unit when `Units` is absent. A number column
without a time unit in `Units` is otherwise never a Duration. `step`, a sample index, carries no
dtype here. On save to a text file the unit of a Duration column resolves in this order:

1. a save keyword of the adopting class, if it has one
2. the `Units` entry of the column in the metadata dict
3. the unit BIDS fixes for a column the adoption declares

Save raises when none gives a unit. The unit used is written to `Units`, so no file ever holds a
duration without its unit. Columns with different `Units` entries are therefore written in
different units. Which unit an adopting class defaults to for its own time columns is the
adoption's choice.

### Conflicts

| situation | outcome |
|---|---|
| keyword on `load` differs from its sidecar entry | keyword wins and replaces the entry in the metadata dict, with a warning, see <a href="#the-metadata-dict">The metadata dict</a> |
| `metadata=` given on `load` | becomes the metadata dict, with `Sources` added when absent, the frame's dtypes come from it, `sidecar` is still the file as read |
| top-level dict that names no column, a free user key or an entry whose column is gone | file-level metadata, kept in both dicts and written, silent |
| top-level key equals a column name, value is not a dict | load and save warn, kept as read, the column keeps its inferred dtype |
| a file exists at `path` or at the sidecar path | save raises unless `overwrite`, the message names the file and the flag |
| direct change to the metadata dict or the frame | not checked until the next save, `sidecar` is stale until then |

The check for existing files runs before anything is written, whatever the files hold. None of
the warnings and raises above depends on `verify_bids`, and each message names the fix. Outside
`verify_bids` a warning or raise means that something would go wrong in pymovements.
`verify_bids` reports BIDS conformity alone.

### Verification

`verify_bids` reports nonconformities at a BIDS level, on save and on load, and is `False` on both
by default. `'REQUIRED'` warns for each required-level finding. `'RECOMMENDED'` warns for
required-level and recommended-level findings. `True` raises on a required-level finding. `False` is
silent. The mechanism runs two required-level checks: the data file is a tsv file with tab
separators, and nulls are written as `n/a` <a href="#references">[2]</a>. A csv file is therefore a
finding when asked and silent otherwise. An adoption declares its own checks, at either level, and
the same `verify_bids` runs them.

### Loading BIDS files

A BIDS-conformant data file loads with the class's `load(path)` alone, whether or not a sidecar
exists next to it, since BIDS recommends the sidecar and does not require it
<a href="#references">[4]</a>. The defaults on load are the BIDS conventions:

- a tsv file is read tab-separated
- `n/a` is read as null
- a column takes its dtype from `Format` where pymovements maps it and from the data otherwise
- a column whose unit BIDS fixes loads as a Duration from that unit, see
  <a href="#duration-columns">Duration columns</a>
- `verify_bids` is `False`

Without a sidecar `sidecar` is `None` and the metadata dict holds what construction writes, the
pymovements stamp and a `Format` per column, and `Sources`.

### Participants and Phenotype adopt

Both classes already model the BIDS sidecar in `metadata`. They adopt with the `schema` values
`participants` and `phenotype`, and every rule above applies to them. What changes for their files
and their signatures is listed under <a href="#backwards-compatibility">Backwards compatibility</a>.
Neither has a time column, `age` carries `years` and stays an integer column.
`MeasurementToolMetadata` in a `Phenotype` sidecar is file-level metadata
<a href="#references">[5]</a> and stays silent.

### Scope

Outside this proposal: `Gaze` and its two YAML files. `BIDSVersion`, `dataset_description.json` with
its `GeneratedBy` list <a href="#references">[8]</a> and BIDS URIs as the `Sources` value form.
Whether `version` stays in the pymovements stamp once `GeneratedBy` is written is decided with that
file. The generic loader that dispatches on `schema`, with its rule for a file labeled for another
class. Reading a sidecar without its data file, and which sidecar a data file reads under a layout,
the inheritance principle <a href="#references">[7]</a>. The data file contract: format selection,
separators, the dtypes a text file can hold, nested columns included, a default time unit, and the
overwrite rule for every file pymovements writes outside this mechanism. The shape of `verify_bids`:
the level and the consequence, warn or raise, as separate parameters, and the result type of the
`verify_bids` method, a verification proposal of its own.

## Rationale

### Why the BIDS shape

BIDS conformity is the target, so the defaults conform and a deviation shows through verification
when asked. The BIDS data dictionary allows file-level keys beside the column entries
<a href="#references">[2]</a>, which is where `Sources` and the `pymovements` entry sit. A probe
with bids-validator 3.0.2 and 2.4.1 <a href="#references">[6]</a>, on a raw and a derivative dataset
with identical results, confirmed it:

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

The validator inspects only the keys it knows, so clean means not looked at for the free key and the
`pymovements` entry. `Sources` it checks in a derivative dataset, as a list of strings, not the form
of the string.

### Why one definition

A rule stated in several places drifts, and two classes already write their sidecars each on their
own terms. The mechanism is defined once, here, and an adoption cites it instead of restating it.

### Why two dicts

`metadata` describes the data object and `sidecar` records the file, and one dict cannot be both.
After load the two differ by design. The file's `Sources` is the file's provenance where the loaded
object's is the file itself. The file's pymovements stamp says which version wrote it where the
metadata dict carries the running one. A load keyword that overrides a sidecar entry is a third
difference. With one dict each difference needed a rule for which side it shows. The other side was
readable only with `json.load`. With two, the metadata dict follows the frame and `sidecar` follows
the file. One correction rule covers every conflict. `Format` cannot be recomputed from the frame,
since `label` and `string` share a dtype. The unit a file was written in exists only in `Units`.
Load therefore carries both into the metadata dict as read. `sidecar` is read-only because it
records a file, and a file changes only through save.

### Why `metadata` carries the pymovements stamp

Construction writes the running pymovements stamp so that the metadata dict reads as the sidecar the
next save would write, pymovements stamp included. The limit is the Conflicts rule: a direct change
to the frame or to the metadata dict shows at the next save. The pymovements stamp warning is where
a preview that did not hold becomes visible. An edited pymovements stamp is replaced at save and the
warning names both values. The pymovements stamp the file carries stays on `sidecar`, where the
version check reads it, so an older file loads silently.

### Why the frame wins at save

The data file is written from the frame, and for a text file `Format` is the only record of how a
column reads back. A `Format` that contradicts the frame would make the next load build a column
that does not match the saved frame. Save replaces it with a warning instead of raising, so a cast
stays a one-step change and the warning says the sidecar followed.

### Why an entry without a column is kept, silently

BIDS treats a description for a nonexistent column as other metadata <a href="#references">[2]</a>,
and the BIDS validator is clean on it. Dropping it would discard something the user wrote, so it
stays in both dicts and is written. It stays without a warning because no warning could be reliable:
BIDS identifies a column entry by its name matching a column <a href="#references">[2]</a>, and that
match is gone with the column. `Format` would mark the entries pymovements wrote, but it is optional
in BIDS and absent from most hand-written entries. And nothing goes wrong in pymovements on such an
entry, so by the rule above it is no warning. A column name whose value is not a dict is different.
The loader looks the dtype up there and finds none, so the column keeps the dtype inferred from the
data. Load and save warn, whatever `verify_bids` says.

### Why verification is opt-in

The aim of the sidecar is that a saved file loads back as it was saved: the frame with its dtypes,
the metadata dict with its entries. The sidecar carries what that takes, `Format` and `Units`, so a
csv file loads back as safely as a tsv file. BIDS conformity is a goal of the caller, not a
condition pymovements enforces, so a nonconformant save is no failure and nothing warns on its own.
The defaults conform, so a caller who writes BIDS files gets them without asking. A caller who wants
the check asks for it with `verify_bids`, on save as on load. What warns or raises without asking is
what would make a file load back differently from how it was saved, or make the record lie, see
<a href="#conflicts">Conflicts</a>.

### Why an existing file raises

pymovements output goes to a derivatives directory, so a file already at `path` or at the sidecar
path was written by someone else or by an earlier run. Writing over a sidecar beside a raw file is
an accident in nearly every workflow, and the lost record does not come back: the next load of that
file degrades to inference. Save therefore raises before it writes anything, and the round trip onto
the same path says `overwrite=True`. One flag governs both files: a flag that protected the sidecar
and not the data beside it would mislead. The data file contract takes the rule over for every other
file pymovements writes. The same check covers two data files that share one stem, since the first
one's sidecar sits at the path. Where none does, nothing is lost, since that file had no record. The
proposal adds no warning for a data file of the same stem beside the target. In the common case, the
same object saved again in another format, that warning would name a file the caller just wrote.
When save raises, the message names such files.

### Why no default time unit

A time unit is a factor on every value of a column, so a wrong one shows nowhere. pymovements
therefore applies a unit only where something fixes it: the `Units` entry, a save keyword of the
adopting class, or the BIDS definition of a column, which the adoption declares. `age` in years is
such a BIDS-fixed unit already <a href="#references">[4]</a>, the released age cap in
`Participants.verify_bids` reads it as years. Where nothing fixes a unit, a number column has none.
Save raises for a Duration column rather than falling back to one. A dataset-wide unit is the
dataset's to state, BIDS does it with an inherited sidecar <a href="#references">[7]</a>. It reaches
the file loader through `metadata=` or the load keyword.

### Why the pymovements stamp

`schema_version` guards the reading pymovements gives a sidecar beyond BIDS: the Duration columns
rule and the keys adoptions add. Every file pymovements writes carries that reading, conformant or
not, since BIDS allows the extra keys <a href="#references">[2]</a>. A third-party sidecar carries
none of it, which is why a file without a pymovements stamp reads silently. `schema` serves a file
outside a BIDS layout, where no file name says which class wrote it. `version` is provenance. BIDS
records it per dataset in `GeneratedBy` <a href="#references">[8]</a>, which a sidecar loaded
outside a layout cannot consult, see <a href="#scope">Scope</a>.

### Alternatives rejected

- A `columns` wrapper for the column dicts. Not the BIDS shape <a href="#references">[2]</a>, and
  the validator no longer finds the descriptions.
- `Sources` inside the `pymovements` entry. The validator checks `Sources` at the top level of a
  derivative sidecar <a href="#references">[6]</a>.
- A schema version per class. One version for the mechanism, since a class adds keys and the
  mechanism defines what a key means.
- A `file` key in the pymovements stamp naming the data file. The clash of two data files on one
  sidecar path is caught by the raise on an existing sidecar, at save, where it arises, and the
  key would be stale after a rename.
- The full file name as the sidecar name, `a.tsv.json`. Not the BIDS name, the validator would
  not find it.
- Recomputing `Format` at every save. Turns `label` into `string`.
- One dict that mirrors the file. Each difference after load needed a rule for which side the one
  dict shows, see <a href="#why-two-dicts">Why two dicts</a>.

## Backwards compatibility

### Participants and Phenotype

What changes from v0.30.0, by where it shows:

| where | what | before | from v0.30.0 |
|---|---|---|---|
| files | a `Format` entry that contradicts the dtype of its column | written as it is | replaced by save, with a warning |
| files | the `Format` of a boolean column | `bool`, which BIDS does not define | `boolean`, load reads both |
| files | a `Format` value pymovements does not map, such as `date` or `uri` | raised | sets no dtype, the column keeps the dtype inferred from the data |
| files | the default sidecar path of `Participants.save` | `participants.json` | `<stem>.json` |
| files | the `pymovements` entry and `Sources` in the sidecar | absent | written, two additional top-level keys for readers of these files |
| files | a file written before v0.30.0 | carries no pymovements stamp | loads as the oldest schema version, silently |
| files | a data file at `path` on save | overwritten | save raises, `overwrite=True` writes over it |
| files | a sidecar at the sidecar path on save | overwritten | save raises, `overwrite=True` writes over it |
| metadata dict | the `pymovements` entry | never held | held from construction on |
| metadata dict | a round trip through save and load | gives the metadata dict back unchanged | adds `Sources` |
| API | the read-only attribute `sidecar` | none | on both classes, additive |
| API | the `verify_bids` default on save | `'REQUIRED'` | `False`, fewer warnings |
| API | the save keyword `overwrite` | none | `overwrite: bool = False`, additive |
| API | the save keyword for the sidecar path | `metadata_path` | `sidecar_path` |
| API | the sidecar path on load | in `metadata` | in `sidecar_path`, `metadata` takes a dict only |
| API | the encoding keyword of the sidecar, save and load | `metadata_encoding` | `sidecar_encoding` |

Under a BIDS layout the participants file is `participants.tsv`, so its default sidecar path does
not change. `metadata_path` on `save`, `metadata_encoding` on `save` and `load`, and a path in
`metadata=` on `load` keep working under a deprecation warning from v0.30.0 and are removed in
v0.35.0. A deprecated keyword given together with its replacement raises. The signatures before,
with the deprecated keywords and after their removal are written out under
<a href="#resulting-signatures">Resulting signatures</a>.

## Implementation

Target release is v0.30.0. One issue per line, drafted once the PMEP is accepted. The module is
private, so every issue is tested through the public methods of the class that has adopted:

- [ ] the shared private module, adopted by `Participants`: the sidecar file shape and style, the
      pymovements stamp and the version rules, the read-only `sidecar` attribute, whose mechanism
      is that issue's, `schema` `participants`, `Sources`, the `<stem>.json` save default.
      Acceptance: a BIDS participants file loads with `Participants.load(path)` alone, with and
      without its sidecar
- [ ] the metadata dict rules and the correction rule: what load builds from `sidecar`, the load
      keyword over an entry, what save corrects, writes and sets
- [ ] `Format` and `Units`: the six values and their dtypes, `boolean` for `bool`, unmapped
      values, the Duration columns rule with the save-time unit order
- [ ] the conflict rules: a column name whose value is not a dict, the entry without a column,
      the raise on an existing data file or sidecar and the `overwrite` keyword
- [ ] `verify_bids`: the `False` default on save, the two required-level checks the mechanism
      runs, and how an adoption declares its own
- [ ] `sidecar_path` and `sidecar_encoding` on save and load, `metadata` on load as a dict only,
      `metadata_path`, `metadata_encoding` and a path in `metadata=` deprecated in v0.30.0 and
      removed in v0.35.0
- [ ] `Phenotype` adopts: `schema` `phenotype`, the pymovements stamp from construction,
      `Sources`, `sidecar`, `MeasurementToolMetadata` stays file-level
- [ ] changelog entry and documentation of the sidecar format

## References

[1] Brain Imaging Data Structure (BIDS) specification v1.11.1, Derivatives, Common data types and
metadata, Common file level metadata fields.
<https://bids-specification.readthedocs.io/en/v1.11.1/derivatives/common-data-types.html#common-file-level-metadata-fields>

[2] Brain Imaging Data Structure (BIDS) specification v1.11.1, Common principles, Tabular files.
<https://bids-specification.readthedocs.io/en/v1.11.1/common-principles.html#tabular-files>

[3] Brain Imaging Data Structure (BIDS) specification v1.11.1, Common principles, Units.
<https://bids-specification.readthedocs.io/en/v1.11.1/common-principles.html#units>

[4] Brain Imaging Data Structure (BIDS) specification v1.11.1, Modality agnostic files, Data summary
files, Participants file.
<https://bids-specification.readthedocs.io/en/v1.11.1/modality-agnostic-files/data-summary-files.html#participants-file>

[5] Brain Imaging Data Structure (BIDS) specification v1.11.1, Modality agnostic files, Phenotypic
and assessment data.
<https://bids-specification.readthedocs.io/en/v1.11.1/modality-agnostic-files/phenotypic-and-assessment-data.html>

[6] bids-validator, the BIDS validator.
<https://github.com/bids-standard/bids-validator>

[7] Brain Imaging Data Structure (BIDS) specification v1.11.1, Common principles, The Inheritance
Principle.
<https://bids-specification.readthedocs.io/en/v1.11.1/common-principles.html#the-inheritance-principle>

[8] Brain Imaging Data Structure (BIDS) specification v1.11.1, Modality agnostic files, Dataset
description, Derived dataset and pipeline description.
<https://bids-specification.readthedocs.io/en/v1.11.1/modality-agnostic-files/dataset-description.html#derived-dataset-and-pipeline-description>

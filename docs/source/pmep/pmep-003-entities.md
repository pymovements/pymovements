# PMEP 3: Entities, entity declarations and path templates

| | |
|---|---|
| **Status** | Draft |
| **Type** | Standards |
| **Author** | Daniel G. Krakowczyk |
| **Created** | 2026-11-18 |
| **Supersedes** | none |

## TL;DR

- Every data object carries `entities` beside `metadata`: what identifies it, kept apart from what
  describes it. A value lives in exactly one of the two, and consumers read identity from
  `entities` only.
- `Gaze`, `Events`, `Participants` and `Phenotype` adopt. An object derived from one source
  object carries its entities.
- An `EntityDefinition` states an entity's name, format and padding. The BIDS entities come
  declared, other keys such as `trial` can be declared. Every write to `entities` is checked.
- `declare_entity` moves a key from `metadata` into `entities`.
- A `PathTemplate` writes the path of a file below a root and parses a path back into entities.
  Without a pattern it follows BIDS and raises on an entity BIDS does not define.
- `save` and `load` take a template. Dataset loaders route `filename_pattern` captures by
  declaration.
- Reading identifiers from `metadata` is deprecated in v0.31.0 and removed in v0.36.0.

## What it looks like

Loading PoTeC, before and after. The call does not change:

```python
dataset = pymovements.Dataset('PoTeC', path='data/PoTeC')
dataset.load()
gaze = dataset.gaze[0]
```

```python
# before (v0.30): what the file name captured sits in metadata beside the rest
gaze.metadata
# {'subject_id': 1, 'text_id': 'b0', 'Sources': ['raw/reader1_b0_raw_data.tsv']}
```

```python
# after: identity in entities, description in metadata
gaze.entities
# {'subject': '01', 'text_id': 'b0'}
gaze.metadata
# {'Sources': ['raw/reader1_b0_raw_data.tsv']}
```

The definition names the captures after their entities and declares what BIDS does not:

```yaml
resources:
  - content: gaze
    filename_pattern: 'reader{subject:d}_{text_id}_raw_data.tsv'
entity_definitions:
  - name: subject      # a BIDS entity, comes declared, the dataset sets the padding
    padding: 2
  - name: text_id      # not a BIDS entity, declared by the dataset
    format: label
```

An object derived from one source object carries its entities:

```python
gaze.detect('ivt')
gaze.events.entities
# {'subject': '01', 'text_id': 'b0'}
```

A single file parses its path into entities, with no dataset:

```python
gaze = pymovements.gaze.from_csv('sub-01_ses-1_task-reading_run-2_physio.tsv')
gaze.entities
# {'subject': '01', 'session': '1', 'task': 'reading', 'run': 2}

phenotype = Phenotype.load('phenotype/handedness.tsv')
phenotype.entities
# {'measurement_tool': 'handedness'}
```

The BIDS template writes BIDS entities only. A custom template writes the others:

```python
PathTemplate().format(gaze.events.entities, suffix='events', extension='tsv')
# ValueError: the BIDS template cannot write the entity 'text_id', which BIDS does not define.
#   Use a template that writes it, or merge by 'text_id' first.

template = PathTemplate('sub-{subject}/sub-{subject}_text-{text_id}_{suffix}.{extension}')
template.format(gaze.events.entities, suffix='events', extension='tsv')
# PurePosixPath('sub-01/sub-01_text-b0_events.tsv')
```

## Resulting signatures

The entities subpackage, `pymovements.entities`, holds the declaration, the store and the
template:

```python
@dataclass(frozen=True)
class EntityDefinition:
    name: str                                   # the key in entities, and the placeholder in
                                                # filename_pattern and in a template pattern
    format: Literal['label', 'index'] = 'label' # label: a string, index: a non-negative integer
    padding: int | None = None                  # minimum number of digits of a value written
                                                # from an integer
    bids_key: str | None = None                 # the key in a BIDS path, such as 'sub', set on
                                                # the BIDS declarations only
```

```python
class Entities(MutableMapping[str, str | int]):
    definitions: Mapping[str, EntityDefinition]   # the declarations in effect, BIDS included
    # every write is checked: an undeclared key raises, a value is cast to its declared format
```

```python
class PathTemplate:
    def __init__(self, pattern: str | None = None) -> None
    # None: the BIDS template, rules from bidsschematools
    # a pattern: '{name}' placeholders for entities, '{datatype}', '{suffix}' and
    #            '{extension}' filled by the class that saves

    def format(
        self,
        entities: Mapping[str, str | int],
        *,
        suffix: str,
        extension: str,
        datatype: str | None = None,
    ) -> PurePosixPath                  # the path below a root, raises on an entity it cannot write

    def parse(self, path: str | PathLike[str]) -> dict[str, str | int]
    # BIDS template: the file name, an empty dict when it is no BIDS name
    # a pattern: the end of the path, as many levels as the pattern has, raises on a miss
```

Every data object that adopts carries the attributes and keywords below, whatever else its
signatures hold. `Gaze`, `Events`, `Participants` and `Phenotype` adopt in this proposal:

```python
entities: Entities                  # what identifies the object
metadata: dict[str, Any]            # what describes it, unchanged

def declare_entity(self, definition: EntityDefinition) -> None
# declares the entity, and moves a key of that name from metadata into entities

def __init__(
    self,
    ...,
    entities: Mapping[str, str | int] | None = None,
    entity_definitions: Sequence[EntityDefinition] | None = None,
) -> None
```

A class with a single-file `save` and `load` takes the template on both. `Phenotype` and the
gaze loaders `pymovements.gaze.from_csv`, `from_ipc`, `from_asc` and `from_begaze` follow it
here, `Events` and later classes with their own `save` and `load`:

```python
def save(
    self,
    path: Path | str | None = None,      # the file, as today
    *,
    root: Path | str | None = None,      # the root the template writes below, instead of path
    template: PathTemplate | None = None,   # None: the BIDS template, used with root only
    ...,
) -> None
# exactly one of path and root

def load(
    path: Path | str,
    *,
    template: PathTemplate | None = None,   # None: the BIDS template
    entity_definitions: Sequence[EntityDefinition] | None = None,
    ...,
) -> Self
```

The dataset definition declares entities, and a resource can fix an entity value for all its
files:

```python
DatasetDefinition(..., entity_definitions: Sequence[EntityDefinition] | None = None)
ResourceDefinition(..., entities: Mapping[str, str | int] | None = None)
```

```yaml
entity_definitions:
  - name: text_id
    format: label
resources:
  - content: precomputed_events
    filename_pattern: 'reader{subject:d}_{text_id}_fixations.tsv'
    entities:
      description: corrected
```

## Motivation

`metadata` holds two kinds of value today. Loading PoTeC puts `subject_id` and `text_id` from
the file name into `metadata`, beside `Sources` and every descriptive key. Nothing marks which
keys say which object this is and which only describe it, so every consumer guesses:

| consumer | what it needs | what it finds today |
|---|---|---|
| data quality report | the subject and session of each object | any key, under any name, as int or string |
| `Gaze.split(by='trial')` | somewhere to put the trial of each part | `metadata['trial']`, beside descriptions |
| saving a dataset | a path for each object | nothing, the caller composes names by hand |
| `Phenotype.load` | the measurement tool the file is about | the file stem, kept nowhere |

The guessing has consequences. A subject loaded as the integer `1` by one loader and the string
`'1'` by another is coerced and merged into one subject. A split trial looks like a description
and is lost when the metadata of two objects is combined. `Sources` gets read as identity when no
key fits, although it records provenance.

BIDS separates the two by construction: entities in the path identify a file, the sidecar
describes it [1, 2]. pymovements saves into BIDS layouts, so its objects need the same separation
in memory and a way to write the path from it and parse it back.

## Specification

### Entities and metadata

Every data object carries `entities` beside `metadata`. `entities` holds what identifies the
object, `metadata` what describes it. A value lives in exactly one of the two. Consumers read
identity from `entities` only, and `Sources` stays the provenance record of PMEP 2, never a
stand-in for a missing entity. The separation is the end state, not a transition step.

### Declarations

An `EntityDefinition` states an entity's name, its format and its padding:

| field | meaning |
|---|---|
| `name` | the key in `entities`, the `filename_pattern` placeholder and the template placeholder |
| `format` | `label`, a string, or `index`, a non-negative integer, as in BIDS [1] |
| `padding` | the minimum number of digits of a value written from an integer, so every template writes the same label for the same entity |
| `bids_key` | the key in a BIDS path, set on the BIDS declarations only |

The BIDS entities come declared under their names in the BIDS schema, read through
bidsschematools [3]: `subject` with the key `sub`, `session`, `task`, `run`, `description` and the
rest. A declaration that names a BIDS entity takes its format and key from the schema and sets
the padding only. Any other key can be declared, such as `trial` or `text_id`. Declarations come
from the dataset definition, from the caller at construction and load, and from `declare_entity`.
There is no global registry.

`subject` is the BIDS name of the entity. The participants file keeps its BIDS column
`participant_id`, whose values are `'sub-' + subject` [4]. That is the join between a
`Participants` table and the objects of its subjects.

### The store

`entities` checks every write against the declarations in effect, whatever writes it: a loader,
a split, `declare_entity` or the caller. An undeclared key raises. A value is cast to its
declared format, and a value that cannot be cast raises. A consumer that meets two objects whose
declarations for the same name differ raises instead of coercing.

### Promotion through declaration

`declare_entity` declares an entity on an existing object. A key of that name in `metadata`
moves into `entities` and leaves `metadata`. There is no way back from `entities` into
`metadata`.

### Loading

Dataset loaders route what `filename_pattern` captures by declaration: declared keys go into
`entities`, cast to their format, the rest into `metadata`. A value a resource fixes in its
`entities` goes into `entities` like a captured one. A captured value that contradicts a fixed
one raises.

`load` and the gaze loaders parse the path they read with the template, the BIDS template by
default. The BIDS template parses the file name, since a BIDS file name repeats every entity of
its path [2]. A name that is not a BIDS name gives no entities, silently. A custom template
parses the end of the path, as many levels as its pattern has, and raises when the path does not
match. A placeholder without a declaration is declared as a label unless `entity_definitions`
declares it. `entities=` on the constructor and on `load` adds to what the path gives, and a
contradicting value raises.

`filename_pattern` stays the way non-BIDS datasets find their files. This proposal changes only
where its captures go.

### Templates

A `PathTemplate` writes the path of a file below a root from the object's entities and parses a
path back into entities. It covers the whole path below the root, directories and file name, and
its entity order is part of it:

| template | writes | entity order | parses |
|---|---|---|---|
| `PathTemplate()` | BIDS entities, directories and name by the BIDS rules [2] | the BIDS order | any BIDS file name |
| `PathTemplate(pattern)` | the placeholders of the pattern | the order of the pattern | the end of a path that matches the pattern |

The rules of the BIDS template come from bidsschematools [3], never from code in pymovements.
The BIDS template writes BIDS entities only. A template raises on an entity it cannot write,
whether BIDS does not define it or the pattern has no placeholder for it, and the message names
the two ways out: a template that writes it, or merging by that entity first. A save with a
template therefore never puts two objects that differ in an entity on one path. A placeholder
without a value raises too. The class that saves fills `datatype`, `suffix` and `extension`.

The template enters `save` through `root`: `save(root=..., template=...)` writes to the path the
template builds below the root. `save(path)` writes exactly that file, as today.

### Adoption

| class | entities |
|---|---|
| `Gaze` | from the loaders, kept while Recording replaces it |
| `Events` | from the loaders and from the gaze object they are detected on |
| `Phenotype` | `measurement_tool`, declared by the class, the stem of `phenotype/<measurement_tool>.tsv` [5] |
| `Participants` | none, `participants.tsv` is dataset-level |

The BIDS template writes and parses the phenotype path by the schema rule for that file, a
template case and no new mechanism. Recording adopts in its own proposal, stimuli later.

### Derived objects

An object derived from one source object carries the source's entities. `Gaze.detect` gives the
events the entities of the gaze. `split` on `Gaze` and `Events` gives each part the entities of
the whole plus the split value, and declares the split key when it is not declared yet: `index`
for an integer column, `label` otherwise. Measures computed from one object carry its entities.
The contract for operations in general belongs to the operation model, adding a split entity on
merge to split and merge by entity.

### Grouping

A consumer that groups by an entity puts objects that lack it into a null group and warns once.
When no object has the entity, that level is left out without a warning.

### Scope

Outside this proposal: scanning a dataset for its files and selecting among them, the dataset
files and selection PMEP. The roots of a dataset and their configuration, the directory layout
PMEP. The file layout `Dataset.save_events` and `Dataset.save_preprocessed` write today, which
`Dataset.load_event_files` reads back, the events layout PMEP. Whether `filename_pattern` merges
with the template. Splitting and merging by entity. A way to move an entity back into `metadata`.

## Rationale

### Why two stores

A marker on some `metadata` keys would leave every consumer filtering one dict and every writer
deciding again which keys count. Two stores make the decision once, at declaration, and a value in
exactly one of them cannot be read both ways. A transition that keeps both filled was rejected:
two copies of an identifier drift as soon as one is edited.

### Why declarations

An identifier's type has to be the same wherever the object came from, or `1` and `'1'` become
two subjects or one by accident. The declaration fixes the type per entity, and the BIDS
declarations fix it as BIDS does: `run` is an index, `subject` a label [1]. Padding sits in the
declaration because it decides the label, so two templates write the same `sub-01` for the same
subject.

### Why the BIDS names

The template takes names, keys and order from the BIDS schema. A second name for the same entity,
such as `participant` or `subject_id`, would need a mapping in every template and every consumer.
`participant_id` is not a second name: it is a column whose values are the subject label with its
prefix [4].

### Why validation sits on the store

Loaders, split, declaration and the caller all write entities. A check on each writer would be
repeated and missed on the next one. A check on the store covers every write, the future ones
included. Declarations are part of the in-memory model, not of file verification, so the
opt-in `verify_bids` of PMEP 2 does not cover them.

### Why the BIDS template raises on other entities

Dropping a non-BIDS entity on save would put every trial of a subject on one path, and the
second write would replace the first. Writing it into a BIDS name would make the file
nonconformant. Raising keeps a default save conformant by construction and names the two ways
out.

### Why the template covers the whole path

BIDS repeats entities in directories and file name, and which directories exist depends on which
entities a file has [2]. Name and directory cannot be cut apart, so the template writes and parses
the path below a root, the same cut `filename_pattern` makes today. Where the roots live is the
directory layout PMEP's question.

### Why load parses by default

A single file has to load with its identity, without a dataset around it. Dataset loaders do all
of that work today, so a single-file workflow loses the entities. A BIDS file name repeats every
entity of its path, so the file name is enough for the default, and no root is needed. Most
non-BIDS names do not parse as BIDS, so a miss of the default is silent. A custom template is the
caller's explicit claim about the layout, so a miss raises.

### Why `root` on save

`save(path)` keeps meaning one file. With the template building the path below `path`, `None` on
save could not mean the BIDS template, and the defaults of `save` and `load` would differ.

### Why bidsschematools

The BIDS rules for entities, their order and directories are maintained in the BIDS schema, and
bidsschematools reads it [3]. Hardcoding them would copy the specification and drift from its
next version.

### Alternatives rejected

- *Identifiers stay in `metadata` with a list of identifying keys.* Every consumer filters, and a
  key can still be read both ways.
- *Resource labels in `metadata`.* PoTeC's corrected and uncorrected fixations differ only in the
  label, and a label that tells files apart is an entity, `description` in BIDS.
- *Dataset names mapped to BIDS keys,* `subject_id` declared with the key `sub`. Two names for one
  entity in every consumer.
- *A separate option for the entity order.* The order is part of the template, the BIDS order for
  the BIDS template and the pattern's for a custom one.
- *A global registry of entity types.* Declarations come from the definition and the caller, the
  BIDS ones from the schema. A registry can follow as an issue.
- *Moving an entity back into `metadata`.* A second way for identity to leave the path. It is
  additive and can follow as an issue if a case comes up.
- *Raising when a grouping meets an object without the entity.* It would block the data quality
  report on common non-BIDS datasets.

## Backwards compatibility

What changes from v0.31.0:

| where | what | before | from v0.31.0 |
|---|---|---|---|
| `metadata` | keys a loader captured from the file name, such as `subject_id` | in `metadata` | in `entities` under the declared name, reading them from `metadata` is deprecated |
| `metadata` | the split value of `Gaze.split` and `Events.split` | `metadata['trial']` | `entities['trial']`, reading it from `metadata` is deprecated |
| built-in definitions | `filename_pattern` placeholders of BIDS entities | `{subject_id:d}` | `{subject:d}` |
| values | an identifier's type | whatever the loader gave | the declared format, `subject` `1` becomes `'01'` with padding 2 |
| `filename_pattern_schema_overrides` | an override for a declared key | sets the type | deprecated, the declaration sets it |
| API | `entities`, `declare_entity`, `entity_definitions`, `PathTemplate`, `EntityDefinition` | none | additive |
| API | `root` and `template` on `save`, `template` on `load` | none | additive |
| dependencies | bidsschematools | none | required |

Reading a moved key from `metadata` returns the entity value with a deprecation warning that
names the new place, from v0.31.0 until its removal in v0.36.0. The value is stored in `entities`
only. A user definition that keeps a placeholder such as `subject_id` still loads, with the value
in `metadata`, and the warning names the rename. The type change of identifiers shows in that
warning and in the migration note.

## Implementation

Target release is v0.31.0. One issue per line, drafted once the PMEP is accepted:

- [ ] `pymovements.entities`: `EntityDefinition`, the BIDS declarations from bidsschematools, the
      `Entities` store and its checks, bidsschematools as a dependency
- [ ] adoption on `Gaze`, `Events`, `Participants` and `Phenotype`: `entities`,
      `entity_definitions`, `declare_entity`, `clone`
- [ ] Dataset loaders: routing captures by declaration, fixed resource values, the built-in
      definitions renamed and declared
- [ ] `PathTemplate`: the BIDS template, custom patterns, `format`, `parse`, the raise on an
      entity a template cannot write
- [ ] `save` and `load` with `root` and `template` on `Phenotype`, `template` on the gaze loaders
- [ ] derived objects: `Gaze.detect`, `split` with its declaration, measures
- [ ] grouping consumers: the null group and its warning, conflicting declarations
- [ ] deprecations: reading moved keys from `metadata`, overrides for declared keys
- [ ] changelog entry and migration note

## References

[1] Brain Imaging Data Structure (BIDS) specification v1.11.1, Appendix, Entities.
<https://bids-specification.readthedocs.io/en/v1.11.1/appendices/entities.html>

[2] Brain Imaging Data Structure (BIDS) specification v1.11.1, Common principles, Filesystem
structure.
<https://bids-specification.readthedocs.io/en/v1.11.1/common-principles.html#filesystem-structure>

[3] bidsschematools, the Python package that reads the BIDS schema.
<https://github.com/bids-standard/bids-specification/tree/master/tools/schemacode>

[4] Brain Imaging Data Structure (BIDS) specification v1.11.1, Modality agnostic files, Data
summary files, Participants file.
<https://bids-specification.readthedocs.io/en/v1.11.1/modality-agnostic-files/data-summary-files.html#participants-file>

[5] Brain Imaging Data Structure (BIDS) specification v1.11.1, Modality agnostic files,
Phenotypic and assessment data.
<https://bids-specification.readthedocs.io/en/v1.11.1/modality-agnostic-files/phenotypic-and-assessment-data.html>

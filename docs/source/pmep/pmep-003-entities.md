# PMEP 3: Entities, entity declarations and path templates

| | |
|---|---|
| **Status** | Draft |
| **Type** | Standards |
| **Author** | Daniel G. Krakowczyk |
| **Created** | 2026-11-18 |
| **Supersedes** | none |

## TL;DR

- Every data object carries `entities` beside `metadata`: what identifies an object is kept apart
  from what describes it. A value lives in exactly one of the two, and consumers read identity
  from `entities` only.
- `Gaze`, `Events`, `Participants` and `Phenotype` adopt. An object derived from one source
  object carries its entities.
- An `EntityDeclaration` states an entity's name, format and padding. The BIDS entities come
  declared, custom keys such as `trial` can be declared. The `entities` store checks each write
  against the declarations: an undeclared key raises, a value is cast to its declared format.
- `declare_entity` moves a key from `metadata` into `entities`.
- A `PathTemplate` builds the relative path of a file from entities and parses a path back into
  entities. Without a pattern it follows the BIDS rules of the installed bidsschematools and
  raises on an entity BIDS does not define.
- `save` and `load` take a template. Dataset loaders route `filename_pattern` captures by
  declaration.
- Keys a loader captured from a file name leave `metadata` in v0.31.0. The migration note names
  the new place.

## What it looks like

A single BIDS file loads with its identity, with no dataset around it. `Events.load` and
`Events.save` are proposed in issue #1563 and land before this proposal's implementation:

```python
events = pymovements.Events.load('sub-01_ses-1_task-reading_run-2_events.tsv')
events.entities
# {'subject': '01', 'session': '1', 'task': 'reading', 'run': 2}
```

The BIDS template builds the relative path back from the entities, directories included, and
names the BIDS version it follows:

```python
template = pymovements.PathTemplate()
template.bids_version
# '1.11.1'
template.build(events.entities, datatype='beh', suffix='events', extension='tsv')
# PurePosixPath('sub-01/ses-1/beh/sub-01_ses-1_task-reading_run-2_events.tsv')
```

`save` and `load` take the template. `save(path)` writes exactly that file, as today, and
`save(root=...)` writes to the path the template builds below the root:

```python
events.save('fixations.tsv')
events.save(root='data/bids')
# data/bids/sub-01/ses-1/beh/sub-01_ses-1_task-reading_run-2_events.tsv
```

A split gives each part the entities of the whole plus the split value, declared on the fly. The
BIDS template writes BIDS entities only, so a part raises on save, and the message names the two
ways out:

```python
trials = events.split(by='trial')
trials[0].entities
# {'subject': '01', 'session': '1', 'task': 'reading', 'run': 2, 'trial': 1}

trials[0].save(root='data/bids')
# ValueError: the BIDS template cannot write the entity 'trial', which BIDS does not define.
#   Use a template that writes it, or combine the objects by 'trial' first.
```

A custom template writes the others. Its placeholders can sit anywhere in the pattern, a
`key-value` form is not required:

```python
custom = pymovements.PathTemplate(
    'sub-{subject}/sub-{subject}_run-{run}_trial-{trial}_{suffix}.{extension}',
)
trials[0].save(root='data/trials', template=custom)
# data/trials/sub-01/sub-01_run-2_trial-1_events.tsv
```

A custom template carries the declarations of its placeholders, and parses a path on load:

```python
template = pymovements.PathTemplate(
    'reader{subject}/reader{subject}_{text_id}_{suffix}.{extension}',
    declarations=[pymovements.EntityDeclaration('text_id')],
)
events = pymovements.Events.load('data/reader1/reader1_b0_fixations.tsv', template=template)
events.entities
# {'subject': '1', 'text_id': 'b0'}
```

Declaring an entity on an existing object moves a key of that name out of `metadata`:

```python
events.metadata
# {'text_id': 'b0', 'Sources': ['data/reader1/reader1_b0_fixations.tsv']}
events.declare_entity(pymovements.EntityDeclaration('text_id'))
events.entities
# {'subject': '1', 'text_id': 'b0'}
events.metadata
# {'Sources': ['data/reader1/reader1_b0_fixations.tsv']}
```

An object derived from one source object carries its entities:

```python
gaze.detect('ivt')
gaze.events.entities == gaze.entities
# True
```

The dataset side does not change its calls. The definition declares what BIDS does not and
names the captures after their entities:

```yaml
entity_declarations:          # declarations, dataset-wide
  - name: subject             # a BIDS entity, comes declared, the dataset sets the padding
    padding: 2
  - name: text_id             # a custom entity, declared by the dataset
    format: label
resources:
  - content: gaze
    filename_pattern: 'reader{subject:d}_{text_id}_raw_data.tsv'
  - content: precomputed_events
    filename_pattern: 'reader{subject:d}_{text_id}_fixations.tsv'
    entities:                 # values, fixed for every file of this resource
      description: corrected
```

```python
dataset.load()
dataset.precomputed_events[0].entities
# {'subject': '01', 'text_id': 'b0', 'description': 'corrected'}
```

## Resulting signatures

The terms, as BIDS defines them [1, 2]:

| term | meaning | in `sub-01_task-reading_run-2_events.tsv` |
|---|---|---|
| entity | a key-value pair in a file name, hyphen-separated | `sub-01`, `task-reading`, `run-2` |
| key | the short form of an entity's name | `sub`, `task`, `run` |
| label | an entity value that is a string of letters and digits | `01`, `reading` |
| index | an entity value that is a non-negative integer, optionally zero-padded | `2` |
| suffix | what follows the last `_` before the extension, names the file's content | `events` |
| extension | what follows the suffix, names the format | `.tsv` |

The entities subpackage, `pymovements.entities`, holds the declaration, the store and the
template:

```python
@dataclass(frozen=True)
class EntityDeclaration:
    name: str                                   # the key in entities, and the placeholder in
                                                # filename_pattern and in a template pattern
    format: Literal['label', 'index'] = 'label' # label: a string, index: a non-negative integer
    padding: int | None = None                  # minimum number of digits when an integer is
                                                # written as a label, 1 becomes '01' with 2
```

```python
class Entities(MutableMapping[str, str | int]):
    declarations: Mapping[str, EntityDeclaration]   # the declarations in effect, BIDS included
    # MutableMapping routes every write through __setitem__, where the check runs:
    # an undeclared key raises, a value is cast to its declared format
```

```python
class PathTemplate:
    def __init__(
        self,
        pattern: str | None = None,             # None: the BIDS template, rules from the
                                                # installed bidsschematools, a pattern: '{name}'
                                                # placeholders for entities, '{datatype}',
                                                # '{suffix}' and '{extension}' filled by the
                                                # class that saves
        *,
        declarations: Sequence[EntityDeclaration] | None = None,
                                                # custom template: declarations of its
                                                # placeholders, an undeclared one is a label
    ) -> None

    bids_version: str                           # the BIDS version of the installed schema

    def build(
        self,
        entities: Mapping[str, str | int],      # the entities of the object
        *,
        suffix: str,                            # the content of the file, such as 'events'
        extension: str,                         # the format of the file, such as 'tsv'
        datatype: str | None = None,            # the BIDS datatype directory, such as 'beh'
    ) -> PurePosixPath                          # the relative path, raises on an entity the
                                                # template cannot write

    def parse(
        self,
        path: str | PathLike[str],              # the path of a file
    ) -> Entities                               # BIDS template: from the file name, empty when
                                                # it is no BIDS name, a pattern: from the end of
                                                # the path, as many levels as the pattern has,
                                                # raises on a miss
```

Every data object that adopts carries the attributes and keywords below, whatever else its
signatures hold. `Gaze`, `Events`, `Participants` and `Phenotype` adopt in this proposal:

```python
entities: Entities                              # what identifies the object
metadata: dict[str, Any]                        # what describes it, the plain dict of PMEP 2

def declare_entity(
    self,
    declaration: EntityDeclaration,             # the entity to declare
) -> None                                       # moves a key of that name from metadata into
                                                # entities

def __init__(
    self,
    ...,
    entities: Entities | Mapping[str, str | int] | None = None,
                                                # the entities, an Entities instance brings its
                                                # declarations along
    entity_declarations: Sequence[EntityDeclaration] | None = None,
                                                # declarations for a plain mapping
) -> None
```

A class with a single-file `save` and `load` takes the template on both. `Events`,
`Participants`, `Phenotype` and the gaze loaders `pymovements.gaze.from_csv`, `from_ipc`,
`from_asc` and `from_begaze` follow it here, later classes with their own `save` and `load`:

```python
def save(
    self,
    path: Path | str | None = None,             # the file, as today
    *,
    root: Path | str | None = None,             # the root the template builds below, instead
                                                # of path
    template: PathTemplate | None = None,       # None: the BIDS template, used with root only
    ...,
) -> None                                       # exactly one of path and root

def load(
    path: Path | str,                           # the file
    *,
    template: PathTemplate | None = None,       # None: the BIDS template, parses the file name
    ...,
) -> Self
```

The dataset definition declares entities, and a resource can fix an entity value for all its
files. `entity_declarations` holds declarations, like the constructor keyword of the same name,
and `entities` holds values, like the attribute of the same name:

```python
DatasetDefinition(..., entity_declarations: Sequence[EntityDeclaration] | None = None)
ResourceDefinition(..., entities: Mapping[str, str | int] | None = None)
```

## Motivation

`metadata` holds two kinds of value today. A dataset loader puts what the file name captured
into `metadata`, beside `Sources` and every descriptive key. Nothing marks which keys identify
the object and which describe it, so every consumer guesses:

| consumer | what it needs | what it finds today |
|---|---|---|
| data quality report | the subject and session of each object | any key, under any name, as int or string |
| `Gaze.split(by='trial')` | somewhere to put the trial of each part | `metadata['trial']`, beside descriptions |
| saving a dataset | a path for each object | nothing, the caller composes names by hand |
| `Phenotype.load` | the measurement tool the file is about | the file stem, kept nowhere |

Three things go wrong. A subject loaded as the integer `1` by one loader and as the string
`'1'` by another is either merged into one subject or kept as two, by accident. A split value
sits beside descriptions, so combining the metadata of two parts treats it as a description and
drops it. `Sources` records provenance, but a consumer that finds no identifying key reads the
file name out of it.

BIDS separates the two by construction: entities in the path identify a file, the sidecar
describes it [1, 2]. pymovements saves into BIDS layouts, so its objects need the same separation
in memory and a way to write the path from it and parse it back.

## Specification

### Entities and metadata

Every data object carries `entities` beside `metadata`. `entities` holds what identifies the
object, `metadata` what describes it. A value lives in exactly one of the two. Consumers read
identity from `entities` only. `Sources` stays the provenance record of PMEP 2, and a consumer
that finds no entity does not read the file name out of it.

### Declarations

An `EntityDeclaration` states an entity's name, its format and its padding:

| field | meaning |
|---|---|
| `name` | the key in `entities`, the `filename_pattern` placeholder and the template placeholder |
| `format` | `label`, a string, or `index`, a non-negative integer, as in BIDS [1] |
| `padding` | the minimum number of digits when an integer is written as a label, so every template writes the same label for the same entity |

The BIDS entities come declared under their names in the BIDS schema, read through
bidsschematools [3]. The schema fixes their names, keys, formats and order:

| entity | key | format | | entity | key | format |
|---|---|---|---|---|---|---|
| `subject` | `sub` | label | | `mtransfer` | `mt` | label |
| `template` | `tpl` | label | | `part` | `part` | label |
| `session` | `ses` | label | | `processing` | `proc` | label |
| `cohort` | `cohort` | label | | `hemisphere` | `hemi` | label |
| `sample` | `sample` | label | | `space` | `space` | label |
| `task` | `task` | label | | `split` | `split` | index |
| `tracksys` | `tracksys` | label | | `recording` | `recording` | label |
| `acquisition` | `acq` | label | | `chunk` | `chunk` | index |
| `nucleus` | `nuc` | label | | `atlas` | `atlas` | label |
| `volume` | `voi` | label | | `segmentation` | `seg` | label |
| `ceagent` | `ce` | label | | `scale` | `scale` | label |
| `tracer` | `trc` | label | | `resolution` | `res` | label |
| `stain` | `stain` | label | | `density` | `den` | label |
| `reconstruction` | `rec` | label | | `label` | `label` | label |
| `direction` | `dir` | label | | `description` | `desc` | label |
| `run` | `run` | index | | | | |
| `modality` | `mod` | label | | | | |
| `echo` | `echo` | index | | | | |
| `flip` | `flip` | index | | | | |
| `inversion` | `inv` | index | | | | |

The table is the one of BIDS 1.11.1, read top-down and left-right. BIDS mandates this order in
every file name [2], so a template that writes or parses a BIDS name needs it. The version in
effect is the one the installed bidsschematools ships, exposed as `bids_version` on the
template, pinned by the dependency lower bound, and a test fails when a bump changes the table.

A declaration that names a BIDS entity takes its format from the schema and sets the padding
only. The key, such as `sub`, is the BIDS template's business and is not part of a declaration.
Any other key can be declared, such as `trial` or `text_id`. Declarations come from the dataset
definition, from the caller at construction, from a custom template and from `declare_entity`.
A registry of entity types beyond that is out of scope, see Scope.

`subject` is the BIDS name of the entity. A participants file keeps its BIDS column
`participant_id`, whose values are `'sub-' + subject` [4], and so does a phenotype file [5].
That column is the join between a `Participants` or `Phenotype` table and the objects of its
subjects. A value with its key prefix is not a second entity, it is the label written the way
the file writes it.

### The store

`entities` checks every write against the declarations in effect, whatever writes it: a loader,
a split, `declare_entity` or the caller. An undeclared key raises. A value is cast to its
declared format, and a value that cannot be cast raises. A consumer that meets two objects whose
declarations for the same name differ, one as a label and one as an index, raises instead of
coercing. Values are `str` or `int`, so there is no narrower type to coerce between.

Validation against declarations is part of the in-memory model. The path a template writes is a
file-level fact, so a class that saves with the BIDS template names "the written path follows
the BIDS rules" as its adoption check, and the opt-in `verify_bids` of PMEP 2 runs it.

### Promotion through declaration

`declare_entity` declares an entity on an existing object. A key of that name in `metadata`
moves into `entities` and leaves `metadata`. Moving an entity back into `metadata` is out of
scope, see Scope.

### Loading

A single file has to load with its identity, without a dataset around it. Dataset loaders do all
of that work today, so a single-file workflow loses the entities.

`load` and the gaze loaders parse the path they read with the template, the BIDS template by
default. The BIDS template parses the file name, since a BIDS file name repeats every entity of
its path [2]. A name that is not a BIDS name gives no entities, silently. A custom template
parses the end of the path, as many levels as its pattern has, and raises when the path does not
match. A placeholder is declared by the template, an undeclared one with the format `label`.
`entities=` on the constructor and on `load` adds to what the path gives, and a contradicting
value raises.

Dataset loaders route what `filename_pattern` captures by declaration: declared keys go into
`entities`, cast to their format, the rest into `metadata`. A value a resource fixes in its
`entities` goes into `entities` like a captured one. A captured value that contradicts a fixed
one raises. `filename_pattern` continues to work as it does, this proposal changes only where
its captures go.

### Templates

A `PathTemplate` builds the relative path of a file from entities and parses a path back into
entities. The relative path covers directories and file name, and its entity order is part of
the template. The template knows no root: the class that saves joins the root and the relative
path, the class that loads hands the template the path it was given.

| template | writes | entity order | parses |
|---|---|---|---|
| `PathTemplate()` | BIDS entities, directories and name by the BIDS rules [2] | the BIDS order | any BIDS file name |
| `PathTemplate(pattern)` | the placeholders of the pattern | the order of the pattern | the end of a path that matches the pattern |

The rules of the BIDS template come from bidsschematools [3], never from code in pymovements.
One BIDS template serves every class: the schema selects the file rule by `datatype` and
`suffix`, which the class that saves fills, so `events` and `physio` under one datatype are two
rules of one template, and the fixed stems `participants.tsv` [4] and
`phenotype/<measurement_tool>.tsv` [5] are two more. A template raises on an entity it cannot
write, whether BIDS does not define it or the pattern has no placeholder for it, and the message
names the two ways out: a template that writes it, or combining the objects by that entity
first. A save with a template therefore never puts two objects that differ in an entity on one
path. A placeholder without a value raises too.

The template enters `save` through `root`: `save(root=..., template=...)` writes to the path the
template builds below the root. `save(path)` writes exactly that file, as today.

### Adoption

| class | entities | BIDS template |
|---|---|---|
| `Gaze` | from the loaders | the `physio` rule |
| `Events` | from the loaders and from the gaze object they are detected on | the `events` rule |
| `Participants` | from the loaders when a resource captures them, otherwise empty | the fixed stem `participants.tsv` [4] |
| `Phenotype` | `measurement_tool`, declared by the class, the stem of `phenotype/<measurement_tool>.tsv` [5] | the `phenotype` rule |

A dataset with one participants file per text captures `text_id` through its resource, and a
custom template such as `participants_text-{text_id}.tsv` writes it back. Combining those files
into one table is out of scope, see Scope.

### Derived objects

An object derived from one source object carries the source's entities. `Gaze.detect` gives the
events the entities of the gaze. `split` on `Gaze` and `Events` gives each part the entities of
the whole plus the split value, and declares the split key when it is not declared yet: `index`
for an integer column, `label` otherwise. Measures computed from one object carry its entities.
Operations that take several sources are out of scope, see Scope.

### Grouping

A consumer that groups by an entity puts objects that lack it into a null group and warns once.
When no object has the entity, that level is left out without a warning.

### Scope

Outside this proposal, with where it goes:

- Scanning a dataset for its files and selecting among them: the dataset files and selection
  PMEP.
- The roots of a dataset and their configuration: the directory layout PMEP.
- The file layout `Dataset.save_events` and `Dataset.save_preprocessed` write today, which
  `Dataset.load_event_files` reads back: the events layout PMEP.
- Whether `filename_pattern` merges with the template: the dataset files and selection PMEP.
- Splitting and merging by entity, including the entities of an object with several sources,
  entities written as columns of a combined table, and combining per-text participants files:
  the split and merge PMEP.
- Splitting an object by a column on save, when the template has an entity the object lacks but
  a column of that name exists: the `Dataset` save methods, issue #1388 and the events layout
  PMEP.
- The BIDS inheritance principle: entities are what a sidecar higher in the tree matches on,
  consolidating sidecars on save and resolving them on load is a roadmap ticket beside the
  directory layout PMEP.
- A registry of entity types, a way to move an entity back into `metadata`, and a custom BIDS
  schema path on the template: additive, each can follow as an issue.

## Rationale

### Why two stores

A marker on some `metadata` keys would leave every consumer filtering one dict and every writer
deciding again which keys count. Two stores make the decision once, at declaration, and a value
in exactly one of them cannot be read both ways. Keeping both stores filled during a transition
would leave two copies of an identifier that drift as soon as one is edited.

### Why declarations

An identifier's type has to be the same wherever the object came from, or `1` and `'1'` become
two subjects or one by accident. The declaration fixes the type per entity, and the BIDS
declarations fix it as BIDS does: `run` is an index, `subject` a label [1]. Padding sits in the
declaration, not in the template, because it decides the label: with it, two templates write the
same `sub-01` for the same subject, with padding per template they would not.

### Why the BIDS names

The template takes names, keys and order from the BIDS schema. A second name for the same entity,
such as `participant` or `subject_id`, would need a mapping in every template and every consumer.
`participant_id` is not a second name: it is a column whose values are the subject label with its
prefix [4]. The order matters because BIDS mandates it, and a template that writes a BIDS name
has to put `sub` before `ses` and `ses` before `task` to produce a valid one [2].

### Why one BIDS template

A BIDS dataset holds many file kinds, but the schema keys its file rules by `datatype` and
`suffix`, and the class that saves knows both. One template object that reads the schema serves
every class, where a template per file kind would repeat the schema lookup in pymovements.

### Why validation sits on the store

Loaders, split, declaration and the caller all write entities. A check in each of those writers
would be repeated, and the next writer would miss it. A check in the store's `__setitem__`
covers every write, the future ones included.

### Why a template raises on an entity it cannot write

Dropping an entity on save would put every trial of a subject on one path, and the second write
would replace the first. Writing a non-BIDS entity into a BIDS name would make the file
nonconformant. Raising keeps a save conformant by construction and names the two ways out. This
holds for any template: the BIDS template cannot write what BIDS does not define, a custom one
cannot write what its pattern has no placeholder for.

### Why the template builds the whole relative path

BIDS repeats entities in directories and file name, and which directories exist depends on which
entities a file has [2]. Name and directory cannot be cut apart, so the template builds the path
below a root, the same cut `filename_pattern` makes today. Where the roots live is out of scope,
see Scope.

### Why load parses by default

A BIDS file name repeats every entity of its path, so the file name is enough for the default,
and no root is needed. Most non-BIDS names do not parse as BIDS, so a miss of the default is
silent. A custom template is the caller's explicit claim about the layout, so a miss raises.
Whether the default stays this way, or `load` parses only when a template is passed and every
miss raises, is open in this draft.

### Why `root` on save

`save(path)` keeps meaning one file. With the template building the path below `path`, `None` on
save could not mean the BIDS template, and the defaults of `save` and `load` would differ.

### Why bidsschematools

The BIDS rules for entities, their order and directories are maintained in the BIDS schema, and
bidsschematools reads it [3]. Hardcoding them would copy the specification and drift from its
next version. One installed pymovements writes one BIDS version, the one its bidsschematools
ships, the same way bids-validator does. The entity table above and the test on it are the
overview pymovements keeps: a bump that changes the table is noticed, not silently adopted.
bidsschematools reads a schema from a directory as well, so a custom schema path on the template
is additive and out of scope.

### Alternatives rejected

- *Identifiers stay in `metadata` with a list of identifying keys.* Every consumer filters, and a
  key can still be read both ways. See "Why two stores".
- *A `Metadata` class that warns on a moved key.* PMEP 2 keeps `metadata` a plain dict, and the
  keys that move are in `metadata` for two releases at most. See "Backwards compatibility".
- *Resource labels in `metadata`.* A label that tells the files of two resources apart, such as
  corrected and uncorrected fixations, identifies them, so it is an entity. See "Entities and
  metadata".
- *Dataset names mapped to BIDS keys,* `subject_id` declared with the key `sub`. Two names for one
  entity in every consumer. See "Why the BIDS names".
- *A `bids_key` field on the declaration.* The BIDS template reads the key from the schema, and
  a field would invite a custom declaration to claim a BIDS key. See "Declarations".
- *A separate option for the entity order.* The order is part of the template, the BIDS order for
  the BIDS template and the pattern's for a custom one. See "Templates".
- *Padding on the template.* Two templates could write two labels for one subject. See "Why
  declarations".
- *A template per file kind.* The schema selects the rule, one template suffices. See "Why one
  BIDS template".
- *Raising when a grouping meets an object without the entity.* A dataset whose files carry no
  session is common, and a consumer that groups by session has to handle it, not refuse it. See
  "Grouping".

## Backwards compatibility

What changes from v0.31.0:

| where | what | before | from v0.31.0 |
|---|---|---|---|
| `metadata` | keys a dataset loader captured from the file name, such as `subject_id` | in `metadata` | in `entities` under the declared name, gone from `metadata`, breaking |
| `metadata` | the split value of `Gaze.split` and `Events.split` | `metadata['trial']` | `entities['trial']`, gone from `metadata`, breaking |
| built-in definitions | `filename_pattern` placeholders of BIDS entities | `{subject_id:d}` | `{subject:d}`, the parse spec stays, the declaration sets the type |
| entity values | an identifier's type | whatever the loader gave | the declared format, `subject` `1` becomes `'01'` with padding 2 |
| `filename_pattern_schema_overrides` | an override for a declared key | sets the type | deprecated, the declaration sets it |
| API | `entities`, `declare_entity`, `entity_declarations`, `PathTemplate`, `EntityDeclaration` | none | additive |
| API | `root` and `template` on `save`, `template` on `load` | none | additive |
| dependencies | bidsschematools | none | required |

The captured keys reached `metadata` with the sources metadata of v0.29.0, only the dataset
loaders write them, and the split value moves with the split rework. A read of a moved key from
`metadata` raises `KeyError`, and the migration note names the new place and the type change.
A user definition that keeps a placeholder such as `subject_id` still loads, with the value in
`metadata`, and the note names the rename.

## Implementation

Target release is v0.31.0. One issue per line, drafted once the PMEP is accepted:

- [ ] `pymovements.entities`: `EntityDeclaration`, the BIDS declarations from bidsschematools, the
      `Entities` store and its checks, bidsschematools as a dependency, the entity table test
- [ ] adoption on `Gaze`, `Events`, `Participants` and `Phenotype`: `entities`,
      `entity_declarations`, `declare_entity`, `clone`
- [ ] Dataset loaders: routing captures by declaration, fixed resource values, the built-in
      definitions renamed and declared
- [ ] `PathTemplate`: the BIDS template with `bids_version`, custom patterns with declarations,
      `build`, `parse`, the raise on an entity a template cannot write
- [ ] `save` and `load` with `root` and `template` on `Events`, `Participants` and `Phenotype`,
      `template` on the gaze loaders, the written path as an adoption check of `verify_bids`
- [ ] derived objects: `Gaze.detect`, `split` with its declaration, measures
- [ ] grouping consumers: the null group and its warning, conflicting declarations
- [ ] deprecation of overrides for declared keys
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

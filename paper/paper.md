---
title: 'pymovements: A Python package for processing eye movement data'
tags:
  - Python
  - eye-tracking
  - psycholinguistics
  - cognitive science
authors:
  - name: Daniel G. Krakowczyk
    orcid: 0009-0009-5100-0733
    affiliation: '1, 2'
    corresponding: true
  - name: David R. Reich
    orcid: 0000-0002-3524-3788
    affiliation: '1, 2'
  - name: Carlson M. Büth
    orcid: 0000-0003-2298-8438
    affiliation: '1'
  - name: Bernhard Angele
    orcid: 0000-0001-8989-8555
    affiliation: '4'
  - name: Jakob Chwastek
    orcid: 0000-0001-7092-6245
    affiliation: '2'
  - name: Deborah N. Jakobi
    orcid: 0000-0002-9719-6673
    affiliation: '1'
  - name: Jana Hofmann
    orcid: 0009-0007-8099-9262
    affiliation: '1'
  - name: Paweł Kasprowski
    orcid: 0000-0002-2090-335X
    affiliation: '5'
  - name: Paul Prasse
    orcid: 0000-0003-1842-3645
    affiliation: '2'
  - name: Andreas Säuberli
    orcid: 0000-0001-9613-334X
    affiliation: '3'
  - name: Anastassia Shaitarova
    orcid: 0000-0003-3124-190X
    affiliation: '1'
  - name: Lena A. Jäger
    orcid: 0000-0001-9018-9713
    affiliation: '1'
affiliations:
  - name: University of Zurich, Switzerland
    index: 1
    ror: 02crff812
  - name: University of Potsdam, Germany
    index: 2
    ror: 03bnmw459
  - name: LMU Munich, Germany
    index: 3
    ror: 05591te55
  - name: Universidad Nebrija, Madrid, Spain
    index: 4
    ror: 03tzyrt94
  - name: Silesian University of Technology, Gliwice, Poland
    index: 5
    ror: 02dyjk442
date: XX XX 2026
bibliography: paper.bib
---

# Summary

Eye movements indicate where a person's visual attention is directed, such
as which word is being read, which part of an image is being inspected, or
which region of a display is being scanned. Eye-tracking devices record
this behavior as a time series of gaze coordinates, and researchers in
psychology, linguistics, neuroscience, and human-computer interaction use
these recordings to study reading, visual attention, oculomotor control,
and interface usability. Turning a raw coordinate stream into
scientifically usable data requires several processing steps: converting
pixel positions into visual angles, smoothing noisy signals, computing
velocities, and segmenting the recording into discrete events such as
fixations, saccades, and blinks.

`pymovements` is an open-source Python package that provides tested,
documented building blocks for such processing pipelines: parsers for
common eye-tracker file formats, including EyeLink's ASC format and SMI
BeGaze exports; a library of preprocessing transformations; established
algorithms for detecting fixations, saccades, and blinks; a collection of
event- and reading-level measures; and functionality for assessing the
data quality of recordings. Plotting utilities support inspection of gaze
data at any stage, from raw signals to detected events. A catalog of
publicly available eye-tracking datasets lets users load each dataset into
a standardized representation without having to harmonize its
idiosyncratic format themselves.

# Statement of need

Eye-tracking preprocessing pipelines are frequently implemented
independently across laboratories, producing inconsistent filtering, event
detection, and data-handling procedures. Even a nominally identical
processing step, such as detecting fixations, can differ substantially
between implementations: an evaluation of ten event-detection algorithms
found systematic disagreement in the events they produce [@Andersson2017].
This variability makes it difficult for researchers in reading and
psycholinguistics, cognitive science, and human-computer interaction to
reproduce or compare analyses across studies. `pymovements`
was introduced to give research groups a shared, tested, and openly
licensed interface for these steps, so that processing is documented,
versioned, and citable rather than re-implemented per project
[@pymovementsPaper]. Widely used event-detection algorithms, including the
velocity-threshold identification (I-VT) and dispersion-threshold
identification (I-DT) methods [@SalvucciGoldberg2000] and the microsaccade
detection algorithm of @EngbertKliegl2003 as implemented in the
Microsaccade Toolbox [@Engbert2015], otherwise exist scattered across
research code bases and publication-specific scripts. `pymovements`
packages them, together with blink detection following @Hershman2018 and
@Nystrom2024, into a single library with a consistent interface.

Two related problems motivated recent additions to the package.
First, there has been no consensus on which properties of a recording setup
or which data-quality metrics should be reported alongside a shared
eye-tracking dataset, a gap the package's data-quality reporting
functionality was developed to address [@Jakobi2024]. Second, existing
public datasets are dispersed across repositories with non-standardized
formats and incomplete metadata. This motivated the package's dataset
catalog, which gives researchers access to more than 35 public
eye-tracking datasets through a single interface and lets dataset authors
contribute their own datasets to increase the visibility of their work
[@Krakowczyk2025].

# State of the field

The eye-tracking tool landscape is broad and fragmented, spanning
recording, visualization, processing, and analysis across many separate
packages [@Niehorster2025]. Several open-source tools address individual
parts of this workflow. In Python, `eyekit` [@eyekit] analyzes reading
behavior over text stimuli, offering areas of interest, reading measures,
and line-assignment correction [@Carr2022], but it operates on
already-detected fixations rather than raw signals. `cili` [@Acland2016]
handles fixation and pupil data, `sideeye` [@Doucette2019] computes reading
measures from fixation reports, `PyTrack` [@GhoseSrinivasan2021] and the
Perception Engineer's Toolbox [@Kubler2020] extract gaze features,
`GazeParser` [@Sogo2019] covers recording and parsing, and dedicated
event-detection algorithms such as I2MC [@I2MC] and REMoDNaV [@REMoDNaV]
classify fixations and saccades from raw samples. In R, `popEye` [@popEye]
spans the path from raw EyeLink files to reading measures, while
`gazeR` [@gazeR] and `eyetrackingR` [@eyetrackingR] process gaze and pupil
data and support the statistical analysis of these signals. Proprietary
vendor software such as SR Research Data Viewer [@DataViewer] provides an
end-to-end but closed-source pipeline tied to specific hardware.

Each of these tools covers a single stage or paradigm: `eyekit` begins
after event detection, the event-detection packages stop at classifying
events, `popEye` is confined to one vendor and to reading, and the others address
recording, feature extraction, statistical analysis, or measures in
isolation. None combines, in a single tested and openly licensed Python
package, the full path from raw vendor files through coordinate transforms
and event detection to reading-level measures, and none offers a
standardized, community-extensible layer for discovering, downloading, and
harmonizing published datasets. `pymovements` was built to fill that gap
rather than to duplicate any single tool: where established algorithms
exist, it reimplements them behind one consistent interface
[@SalvucciGoldberg2000; @EngbertKliegl2003; @Carr2022] rather than
replacing them, and it adds the dataset-harmonization layer that has no
counterpart among them.

# Software design

`pymovements` is organized around a single in-memory representation, the
`Gaze` object, which stores raw samples together with the experiment
metadata (screen geometry, sampling rate, and eye-tracker parameters) that
downstream operations require, as well as any detected events. Keeping the
signal and its acquisition context together lets steps such as
pixel-to-degree conversion be expressed without the user re-supplying
calibration parameters at each call. The object is backed by the `polars`
dataframe library [@polars], whose columnar, Rust-based engine provides the
throughput needed to process large raw gaze recordings; this trades some
familiarity for performance, as most researchers know the `pandas`
ecosystem better. Constructors from `pandas` dataframes [@pandas] and
`numpy` arrays [@numpy] mitigate this cost and keep the package
interoperable with these widely used libraries. At load time, the
package validates a dataset's declared configuration against the data and
surfaces misconfigurations as structured reports instead of silent errors
deeper in the pipeline.

Extensibility is a primary design goal. Preprocessing transforms,
event-detection algorithms, and measures are each exposed through
registries so that a research group can add and dispatch its own methods
without modifying the package. Public datasets are described declaratively
as dataset definitions that point at the resources published by each
dataset's original authors rather than redistributing any data; this
keeps the catalog on firm legal and ethical ground as it grows, lets
definitions be contributed as lightweight YAML files by researchers without
deep Python experience, and is backed by automated checks that the
referenced resources remain reachable and that downloaded files match their
published checksums. The package aligns with community data standards to
keep its outputs findable, accessible, interoperable, and reusable
[@Wilkinson2016]: participant and phenotype data follow the metadata
conventions of the Brain Imaging Data Structure (BIDS) [@Gorgolewski2016],
and data-quality reports are written as BIDS derivatives.

The package follows established open-development practices: an automated
test suite with 100% code coverage runs under continuous integration;
every change requires a line-by-line review by a maintainer before merging;
a contributing guide documents the workflow for new contributors; and the
documentation, with tutorials and a full API reference, is hosted on Read
the Docs. The substantial engineering effort of the project lies in this
implementation and its coherent integration of otherwise disparate methods,
not in any single new algorithm.

# Research impact statement

`pymovements` has been in active development since 2022 and has been the
subject of four publications: three peer-reviewed papers at the ACM
Symposium on Eye Tracking Research and Applications [@pymovementsPaper;
@Jakobi2024; @Krakowczyk2025] and one at the MultiplEYE Final Conference
[@Krakowczyk2026]. It serves as the main backend of the data-preprocessing
pipeline of the MultiplEYE COST Action (CA21131), a pan-European
reading-research consortium, with continuous exchange between the two
projects; four multi-day contributor meetings organized through the COST
Action brought together contributors from across Europe to jointly design
and implement core features of the package [@Krakowczyk2026]. `pymovements`
has more than 50 contributors from multiple laboratories and is distributed
via PyPI and conda-forge, with more than 90,000 downloads from PyPI
[@pepyPymovements] and more than 58,000 from conda-forge
[@condaforgePymovements] as of August 2026.

Beyond the author team, `pymovements` has been taken up in research across
several domains, including gaze data-quality research [@Elfares2025;
@Roehrl2026], reading and language-model research [@LopezCardona2025;
@LuGe2026], clinical vision research using virtual-reality simulations
[@Grootjen2025; @Grootjen2025b; @Grootjen2025c], human-computer interaction
[@Chiossi2024; @Chiossi2024b], automotive and applied ergonomics
[@Lopez2025], and virtual-reality interaction [@Li2024; @Li2025;
@Wang2026]. It is also cited as a reference point by newer eye-tracking
analysis tools [@PyNeon2026; @OpenGazeLab2026; @Balaskas2026] and featured
in a recent review of eye-tracking software [@Niehorster2025]. Its dataset
catalog has begun to receive contributions from researchers outside
the core team [@MCFW-Gaze-Paper], supported by a dedicated dataset
contribution guide that lowers the barrier to adding new datasets.

# AI usage disclosure

Generative AI tools were used, under human direction, in developing both
the software and this paper. Anthropic's Claude, JetBrains' Junie, and
GitHub Copilot assisted some contributors with code generation and review,
and Claude was used to draft and copy-edit this paper. Every change to the
codebase requires an approving line-by-line review from at least one
maintainer before merging, enforced by the repository's branch protection,
so no AI-assisted contribution reaches the released package without human
review. All design decisions were made by the human authors, who reviewed
and validated the AI-assisted outputs in this paper, verified every
reference against its cited source, and take responsibility for its
content.

# Acknowledgements

The pymovements project was partially funded by the Swiss National Science
Foundation (SNSF) under grants IZCOZ0_220330 (EyeNLG) and 212276 (MeRID),
and the German Federal Ministry of Education and Research (BMBF) under grant
01IS20043. It was further supported by the COST
Action MultiplEYE (CA21131) of the European Cooperation in Science and
Technology (COST). The funding bodies had no role in the design of the
software or in the writing of this paper.

# References

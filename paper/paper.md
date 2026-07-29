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
    affiliation: "1, 2"
    corresponding: true
  - name: David R. Reich
    orcid: 0000-0002-3524-3788
    affiliation: "1, 2"
  - name: Carlson Moses Büth
    orcid: 0000-0003-2298-8438
    affiliation: 1
  - name: Paul Prasse
    orcid: 0000-0003-1842-3645
    affiliation: 2
  - name: Andreas Säuberli
    orcid: 0000-0001-9613-334X
    affiliation: 3
  - name: Deborah N. Jakobi
    orcid: 0000-0002-9719-6673
    affiliation: 1
  - name: Jakob Chwastek
    orcid: 0000-0001-7092-6245
    affiliation: 2
  - name: Anastassia Shaitarova
    orcid: 0000-0003-3124-190X
    affiliation: 1
  - name: Bernhard Angele
    orcid: 0000-0001-8989-8555
    affiliation: 4
  - name: Paweł Kasprowski
    orcid: 0000-0002-2090-335X
    affiliation: 5
  - name: Lena A. Jäger
    orcid: 0000-0001-9018-9713
    affiliation: 1
affiliations:
  - name: University of Zurich, Switzerland
    index: 1
  - name: University of Potsdam, Germany
    index: 2
  - name: Ludwig Maximilian University of Munich, Germany
    index: 3
  - name: Universidad Nebrija, Madrid, Spain
    index: 4
  - name: Silesian University of Technology, Gliwice, Poland
    index: 5
date:
bibliography: paper.bib
---

# Summary

Eye movements indicate where a person's visual attention is directed, such
as which word is being read, which part of an image is being inspected, or
which region of a display is being scanned. Eye-tracking devices record
this behavior as a time series of gaze coordinates, and researchers in
psychology, linguistics, neuroscience, and human-computer interaction use
these recordings to study reading, visual attention, oculomotor control,
and the usability of user interfaces. Turning a raw coordinate stream into
scientifically usable data requires several processing steps: converting
pixel positions into visual angles, smoothing noisy signals, computing
velocities, and segmenting the recording into discrete events such as
fixations, saccades, and blinks.

`pymovements` is an open-source Python package that provides tested,
documented building blocks for such processing pipelines, including parsers
for common eye-tracker file formats, a library of preprocessing
transformations, established algorithms for detecting fixations, saccades,
and blinks, a collection of event- and reading-level measures, and plotting
utilities for inspecting the resulting data. It also includes a catalog of
publicly available eye-tracking datasets that users can download and load
into a standardized representation, without having to harmonize each
dataset's idiosyncratic format themselves.

# Statement of Need

Eye-tracking researchers often rely on a range of lab-specific scripts and
single-purpose tools to preprocess gaze data, several of which are
published without systematic test coverage or ongoing maintenance
[@Acland2016; @Doucette2019; @GhoseSrinivasan2021; @Kubler2020; @Sogo2019].
This makes it difficult to reproduce or compare analyses across studies,
since a nominally identical processing step, such as detecting fixations,
can differ between implementations without either author being aware of it.
`pymovements` was introduced to give research groups a shared, tested, and
openly licensed interface for these steps, so that processing is
documented, versioned, and citable rather than re-implemented per project
[@pymovementsPaper]. Event detection algorithms such as the
velocity-threshold identification (I-VT) and dispersion-threshold
identification (I-DT) methods [@SalvucciGoldberg2000], and the microsaccade
detection algorithm of [@EngbertKliegl2003] as implemented in the
Microsaccade Toolbox [@Engbert2015], are widely used but exist scattered
across research code bases and publication-specific scripts. `pymovements`
packages these algorithms, together with blink detection following
[@Hershman2018; @Nystrom2024], into that single tested library.

Two further, related problems motivated recent additions to the package.
First, there has been no consensus on which properties of a recording setup
or which data-quality metrics should be reported alongside a shared
eye-tracking dataset, which `pymovements`' data-quality reporting
functionality was developed to address [@Jakobi2024]. Second, existing
public datasets are scattered across repositories with non-standardized
formats and incomplete metadata, motivating the package's dataset library,
which lets researchers download and load more than 35 public eye-tracking
datasets through a single interface, and lets dataset authors contribute
their own dataset definitions to increase visibility of their work
[@Krakowczyk2025].

`pymovements` has been in active development since 2022 and has been the
subject of four peer-reviewed papers: three at the ACM Symposium on Eye
Tracking Research and Applications (ETRA) [@pymovementsPaper; @Jakobi2024;
@Krakowczyk2025] and one at the MultiplEYE Final Conference [@Krakowczyk2026].
It is developed collaboratively by more than 50 contributors across multiple
research laboratories. `pymovements` is central to the MultiplEYE COST Action
(CA21131), serving as the main backend of its data preprocessing pipeline,
with continuous exchange between the two projects [@Krakowczyk2026]. Through
the COST Action, which provided funding, four multi-day contributor meetings
brought together contributors from across Europe to jointly design and
implement core features of the package. The
dataset library has also begun to attract contributions from researchers
outside the core team, supported by a dedicated dataset contribution guide
that lowers the barrier to adding new datasets and encourages the community
to share their own.

# Functionality

The central data structure of `pymovements` is the `Gaze` object,
backed by the `polars` dataframe library [@polars], which holds raw
gaze samples alongside experiment and eye-tracker metadata. Around
this data structure, the package provides:

- **Dataset library**: definitions for dozens of publicly available
  eye-tracking datasets (reading corpora, free-viewing, and other
  paradigms), each with automatic download, checksum verification, and
  loading into a unified representation via the `Dataset` API.
  `pymovements` does not host or redistribute any dataset resources itself;
  it only points to and downloads the resources published by each dataset's
  original authors, to whom users are instructed to give credit. Dataset
  definitions can be contributed by their original authors to increase the
  visibility of their published datasets [@Krakowczyk2025].
- **BIDS-compliant metadata**, reflecting a recent focus on interoperability
  with established data standards: `pymovements` supports loading and
  validating participant data as well as phenotype and assessment data
  following the Brain Imaging Data Structure (BIDS) specification
  [@Gorgolewski2016].
- **Readers** for common formats, including EyeLink ASCII (`.asc`), BeGaze,
  CSV, and Arrow IPC files, as well as construction from existing `numpy`
  or `pandas` data via `from_numpy`/`from_pandas`.
- **Preprocessing transforms**, including pixel-to-degrees and
  degrees-to-pixels conversion, position-to-velocity and
  position-to-acceleration computation, resampling, smoothing (including
  Savitzky-Golay filtering, [@SavitzkyGolay1964]), clipping, and
  normalization.
- **Event detection**, including the I-VT and I-DT algorithms
  [@SalvucciGoldberg2000], the microsaccades algorithm [@EngbertKliegl2003;
  @Engbert2015], blink detection [@Hershman2018; @Nystrom2024], and
  detection of samples falling outside the screen area, all exposed through
  an extensible `EventDetectionLibrary`.
- **Fixation drift correction**, offering eleven automated vertical
  drift-correction algorithms for multi-line reading data, together with a
  "Wisdom of the Crowd" ensemble method that combines their predictions by
  majority vote, following the taxonomy of [@Carr2022].
- **Measures**, computed either per event (e.g. amplitude, duration, peak
  velocity) or per sample (e.g. data loss, null ratio), as well as a
  dedicated set of reading measures for use in psycholinguistic reading
  studies, including the proportion of fixations (and fixation duration)
  falling outside any defined area of interest, a data-quality indicator of
  whether drift correction is needed. Areas of interest can be defined over
  text and image stimuli.
- **Data validation and quality reports**, cross-checking a dataset's
  declared configuration (columns, dtypes, gaze components) against the
  actual schema of the loaded data at load time, and surfacing the result
  as a structured, human-readable data-quality report following the
  reporting standards proposed in [@Jakobi2024].
- **Plotting**, including scanpath plots, gaze heatmaps, time-series
  traces, main-sequence plots, and data-loss histograms, for visual
  inspection of raw data and detected events.

The package is tested with a continuous integration suite, distributed via
PyPI and conda-forge, and documented with a user guide, tutorials, and a
full API reference.

# Conflicts of Interest

The authors declare that they have no competing interests. The funding
bodies listed in the Acknowledgements had no role in the design of the
software or in the writing of this paper.

# Acknowledgements

The pymovements project was partially funded by the Swiss National Science
Foundation (SNSF) under grants IZCOZ0_220330 (EyeNLG) and 212276 (MeRID),
the German Federal Ministry of Education and Research (BMBF) under grant
01IS20043, the DAAD programme Konrad Zuse Schools of Excellence in
Artificial Intelligence (ELIZA), and the Romanian National Research Council
(CNCS) through the Executive Agency for Higher Education, Research,
Development and Innovation Funding (UEFISCDI) under grant
PN-IV-P2-2.1-TE-2023-2007 (InstRead). It was further supported by work from
European Cooperation in Science and Technology (COST) and the COST Action
MultiplEYE (CA21131).

# References

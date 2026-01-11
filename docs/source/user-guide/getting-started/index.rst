================
Getting Started
================

Whether you are working with eye-tracking data for the first time in a student assignment,
routinely collecting and analysing reading data in the lab, or reusing and/or publishing
datasets as part of a larger research effort, working with eye-tracking data often begins in the
same way. You are faced with files exported from an eye tracker, often accompanied by stimuli,
logs, and metadata, and the task of turning these heterogeneous files into data that are
meaningful, reliable, and reusable.

This guide introduces the concepts and pymovements workflows, data structures, and functions needed
to move from raw eye-tracker recordings to analysis-ready and reusable data.


What is eye-tracking data?
--------------------------

Eye-tracking data consists of measurements of eye position over time, typically recorded at a fixed
sampling frequency.

An eye tracker estimates the point of gaze, that is, where on a stimulus or display a participant
is inferred to be looking, by measuring the relative position of the pupil and corneal reflections.
During calibration, participants fixate known reference points, allowing the system to learn a
mapping from eye position signals to gaze coordinates on the stimulus.
Crucially, eye-tracking data are signals rather than direct measurements of perception or
cognition. Constructs such as attention, comprehension, or cognitive processes are inferred through
preprocessing, event detection, and analysis choices.

Depending on the experimental setup, participants may be:

- reading texts,
- viewing static images,
- watching videos,
- or interacting with dynamic or real-world stimuli.

At the most basic level, eye-tracking data consist of **time-ordered gaze samples** which typically
include:

- a timestamp,
- horizontal and vertical gaze coordinates,
- optional pupil size estimates,
- and device- or vendor-specific fields.

In screen-based experiments, gaze coordinates are commonly expressed in pixel units, corresponding
to positions on the display surface.

Eye-tracking analysis as a sequence of transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Recorded data**: Vendor-specific files produced by the eye tracker and experiment software.
2. **Raw samples**: Time-ordered gaze measurements extracted from the recordings.
3. **Preprocessed samples**: Samples that have been cleaned, filtered, transformed into meaningful
    units, or restricted to relevant time windows (e.g. trials).
4. **Events**: Segments of the signal classified as fixations, saccades, blinks, or other
    eye-movement events using detection algorithms.
5. **Analysis measures**: Summary statistics, models, or visualizations derived from samples or
    events.

It is important to keep in mind that there is no single preprocessing pipeline or eye-tracking
measure that fits all research questions; choices depend on the research goal and data quality.
Each data transformation introduces assumptions, and making these visible is essential for valid
and reproducible analysis. Data quality plays a central role throughout this process. Issues such
as calibration errors, data loss, or unstable sampling can propagate through multiple processing
steps and substantially affect results.

In the following sections, you will learn how to:

- understand the structure and meaning of eye-tracking data,
- inspect and evaluate data quality,
- preprocess raw samples,
- detect and evaluate oculomotoric events,
- visualize eye movements and summarize behavior,
- work with standardized datasets and metadata,
- prepare, validate, and publish reusable eye-tracking datasets.

You can think of this guide as moving from signals → structure → interpretation → reuse.

Installing pymovements


.. toctree::
   :maxdepth: 1

    installation
